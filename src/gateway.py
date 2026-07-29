"""Nano-Dynamo: two-phase disaggregated proxy with KV-aware routing.

PHASE 1: Disaggregated Prefill/Decode — route to independent P/D GPU pools.
PHASE 2: KV-Aware Routing — route based on prefix cache overlap + load.
PHASE 3: KVBM — KV Block Migration (track placement + decode selection)
PHASE 4: Preemption & Promotion — LRU eviction, session promotion
PHASE 5: Dynamic Pool Scaling — worker drain, rebalance, auto-scale
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .kv_events import token_ids_to_block_hashes
from .kv_index import KvIndex, OverlapResult
from .kvbm import MigrationManager
from .preemption import PreemptionManager
from .scaling import ScalingManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("nano-dynamo")

HTTP_TIMEOUT = httpx.Timeout(600.0, connect=10.0)

_tokenizer = None
_tokenizer_lock = threading.Lock()


def _get_tokenizer(model_name: str):
    global _tokenizer
    if _tokenizer is None:
        with _tokenizer_lock:
            if _tokenizer is None:
                from transformers import AutoTokenizer
                _tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return _tokenizer


# ── Scheduling policy (Phase 2: KV-aware) ──────────────────────────

@dataclass
class ScheduleContext:
    token_ids: List[int]
    overlap: OverlapResult
    worker_loads: Dict[int, int]

class SchedulingPolicy(ABC):
    @abstractmethod
    def select(self, num_workers: int, ctx: Optional[ScheduleContext] = None) -> int:
        ...

class RoundRobinPolicy(SchedulingPolicy):
    def __init__(self):
        self._counter = 0

    def select(self, num_workers: int, ctx: Optional[ScheduleContext] = None) -> int:
        idx = self._counter % num_workers
        self._counter += 1
        return idx

class KvAwarePolicy(SchedulingPolicy):
    """Route to the prefill worker with highest KV cache overlap.

    Falls back to round-robin when no overlap data is available.
    """
    def __init__(self):
        self._rr = RoundRobinPolicy()

    def select(self, num_workers: int, ctx: Optional[ScheduleContext] = None) -> int:
        if ctx is None or not ctx.overlap.scores:
            return self._rr.select(num_workers)

        candidates = [(wid, score) for wid, score in ctx.overlap.scores.items() if wid < num_workers]
        if not candidates:
            return self._rr.select(num_workers)

        best_idx, best_score = max(candidates, key=lambda x: x[1])
        log.info("KV_ROUTE[overlap] worker=%d overlap_blocks=%d isl=%d",
                 best_idx, best_score, len(ctx.token_ids))
        return best_idx


# ── Two-phase proxy ────────────────────────────────────────────────

class DisaggProxy:
    def __init__(
        self,
        prefill_instances: list[str],
        decode_instances: list[str],
        model: str,
        mode: str = "push",
        prefill_engine_ids: Optional[list[str]] = None,
        prefill_kv_host: str = "127.0.0.1",
        prefill_side_channel_ports: Optional[list[int]] = None,
        prefill_tp_size: int = 1,
        prefill_pp_size: int = 1,
        policy: Optional[SchedulingPolicy] = None,
        block_size: int = 16,
        decode_cache_capacity: int = 4096,
        migration_threshold: float = 0.80,
        preempt_threshold: float = 0.85,
        preempt_batch: int = 1,
    ):
        self.prefill_instances = prefill_instances
        self.decode_instances = decode_instances
        self.model = model
        self.mode = mode
        self.block_size = block_size
        self.policy = policy or KvAwarePolicy()
        self.preempt_threshold = preempt_threshold
        self.preempt_batch = preempt_batch

        self.p_engine_ids = prefill_engine_ids or [f"prefill-{i}" for i in range(len(prefill_instances))]
        self.p_side_ports = prefill_side_channel_ports or [5600 + i for i in range(len(prefill_instances))]
        self.p_kv_host = prefill_kv_host
        self.p_tp_size = prefill_tp_size
        self.p_pp_size = prefill_pp_size

        self.kv_index = KvIndex()
        self.tokenizer = _get_tokenizer(model)

        # Phase 3: KV Block Migration manager
        self.migration = MigrationManager(
            prefill_instances=prefill_instances,
            decode_instances=decode_instances,
            decode_cache_capacity=decode_cache_capacity,
            migration_threshold=migration_threshold,
        )

        # Phase 4: Preemption manager
        self.preemptor = PreemptionManager(self.migration)

        # Phase 5: Dynamic pool scaling
        self.scaler = ScalingManager()

        # Track conversation_id -> (p_idx, d_idx) for multi-turn cache locality
        self._conv_worker: Dict[str, tuple] = {}

        log.info(
            "DisaggProxy mode=%s model=%s P=%d D=%d policy=%s",
            mode, model, len(prefill_instances), len(decode_instances),
            type(policy).__name__,
        )

    def _tokenize(self, body: dict) -> List[int]:
        messages = body.get("messages", [])
        prompt = body.get("prompt", "")
        if messages:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = prompt
        return self.tokenizer.encode(text) if text else []

    def _pick(self, body: dict) -> tuple[str, str, int, str, int, int]:
        token_ids = self._tokenize(body)
        conv_id = body.get("conversation_id")

        # Phase 4: Promotion — check if this was a preempted session
        if conv_id:
            promoted = self.preemptor.promote(conv_id)
            if promoted is not None:
                d_overlap = self.migration.registry.overlap_decode(
                    promoted.original_d_idx, promoted.block_hashes,
                )
                if d_overlap > 0 and not self.scaler.is_draining_decode(promoted.original_d_idx):
                    log.info("PROMOTE[%s] original D%d overlap=%d",
                             conv_id[:8], promoted.original_d_idx, d_overlap)
                    d_idx = promoted.original_d_idx
                    p_idx = promoted.original_p_idx
                    self._conv_worker[conv_id] = (p_idx, d_idx)
                    p_url = self.prefill_instances[p_idx]
                    d_url = self.decode_instances[d_idx]
                    eid = self.p_engine_ids[p_idx]
                    sport = self.p_side_ports[p_idx]
                    return p_url, d_url, p_idx, eid, sport, d_idx
                log.info("PROMOTE[%s] blocks evicted or draining, re-routing", conv_id[:8])
            else:
                self.preemptor.access(conv_id)

        # Phase 2+3+5: Standard routing — skip draining workers
        p_valid = [i for i in range(len(self.prefill_instances))
                   if not self.scaler.is_draining_prefill(i)]
        d_valid = [i for i in range(len(self.decode_instances))
                   if not self.scaler.is_draining_decode(i)]

        if conv_id and conv_id in self._conv_worker:
            p_idx, d_idx = self._conv_worker[conv_id]
            if (p_idx in p_valid and d_idx in d_valid):
                log.debug("CONV_CACHE[%s] reuse P%d D%d", conv_id[:8], p_idx, d_idx)
                p_url = self.prefill_instances[p_idx]
                d_url = self.decode_instances[d_idx]
                eid = self.p_engine_ids[p_idx]
                sport = self.p_side_ports[p_idx]
                return p_url, d_url, p_idx, eid, sport, d_idx

        if not p_valid:
            p_valid = list(range(len(self.prefill_instances)))
        if not d_valid:
            d_valid = list(range(len(self.decode_instances)))

        # Phase 2: overlap-based prefill selection among valid workers
        all_scores = self.kv_index.compute_overlap(token_ids, self.block_size).scores
        p_scores = {i: all_scores.get(i, 0) for i in p_valid}
        ctx = ScheduleContext(token_ids=token_ids,
                              overlap=OverlapResult(scores=p_scores),
                              worker_loads={})
        idx = self.policy.select(len(self.prefill_instances), ctx)
        if idx not in p_valid:
            idx = p_valid[0]

        # Phase 3+5: KVBM decode selection among valid workers
        block_hashes = set(token_ids_to_block_hashes(token_ids, self.block_size))
        d_idx, d_overlap = self.migration.best_decode(block_hashes, valid_only=d_valid)

        p_url = self.prefill_instances[idx]
        d_url = self.decode_instances[d_idx]
        eid = self.p_engine_ids[idx]
        sport = self.p_side_ports[idx]

        log.info("KVBM route P%d D%d overlap=%d", idx, d_idx, d_overlap)

        if conv_id:
            self._conv_worker[conv_id] = (idx, d_idx)

        return p_url, d_url, idx, eid, sport, d_idx

    def _record_blocks(self, p_idx: int, d_idx: int, token_ids: List[int], conv_id: Optional[str] = None):
        hashes = token_ids_to_block_hashes(token_ids, self.block_size)
        if hashes:
            self.kv_index.record_blocks(p_idx, hashes)
            self.migration.record_prefill_blocks(p_idx, hashes)
            self.migration.record_decode_blocks(d_idx, hashes)
            if conv_id:
                self.preemptor.register(conv_id, token_ids, hashes, p_idx, d_idx)

    async def _forward_stream(self, url: str, body: dict, headers: dict) -> AsyncGenerator[bytes, None]:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as c:
            async with c.stream("POST", url, json=body, headers=headers) as resp:
                if resp.status_code != 200:
                    err = await resp.aread()
                    raise RuntimeError(
                        f"Upstream {url} returned {resp.status_code}: {err[:200].decode(errors='replace')}"
                    )
                async for chunk in resp.aiter_bytes():
                    yield chunk

    async def _forward_once(self, url: str, body: dict, headers: dict) -> httpx.Response:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as c:
            return await c.post(url, json=body, headers=headers)

    async def _drain(self, url: str, body: dict, headers: dict):
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as c:
            async with c.stream("POST", url, json=body, headers=headers) as resp:
                async for _ in resp.aiter_bytes():
                    pass

    async def _push(self, raw_request: Request):
        body = await raw_request.json()
        req_id = str(uuid.uuid4())
        p_url, d_url, idx, eid, sport, d_idx = self._pick(body)
        headers = {"X-Request-Id": req_id}

        p_body = body.copy()
        p_body["max_tokens"] = 1
        p_body["kv_transfer_params"] = {
            "do_remote_decode": True,
            "do_remote_prefill": False,
            "remote_engine_id": None,
            "remote_block_ids": None,
            "remote_host": None,
            "remote_port": None,
        }

        d_body = body.copy()
        d_body["kv_transfer_params"] = {
            "do_remote_decode": False,
            "do_remote_prefill": True,
            "remote_engine_id": eid,
            "remote_host": self.p_kv_host,
            "remote_port": sport,
            "tp_size": self.p_tp_size,
            "pp_size": self.p_pp_size,
            "remote_request_id": req_id,
        }

        log.info("PUSH[%s] P=%s D=%s engine=%s", req_id[:8], p_url, d_url, eid)

        token_ids = self._tokenize(body)
        p_task = asyncio.create_task(
            self._drain(f"http://{p_url}/v1/chat/completions", p_body, headers)
        )

        try:
            d_stream = self._forward_stream(f"http://{d_url}/v1/chat/completions", d_body, headers)
        except RuntimeError as e:
            p_task.cancel()
            log.error("Decode-side failure for PUSH[%s]: %s", req_id[:8], e)
            return JSONResponse(status_code=502, content={"error": str(e)})

        await p_task
        self._record_blocks(idx, d_idx, token_ids, body.get("conversation_id"))

        return StreamingResponse(d_stream, media_type="text/event-stream")

    async def _pull(self, raw_request: Request):
        body = await raw_request.json()
        req_id = str(uuid.uuid4())
        p_url, d_url, idx, eid, sport, d_idx = self._pick(body)
        headers = {"X-Request-Id": req_id}

        p_body = body.copy()
        p_body["max_tokens"] = 1
        p_body["kv_transfer_params"] = {
            "do_remote_decode": True,
            "do_remote_prefill": False,
            "remote_engine_id": None,
            "remote_block_ids": None,
            "remote_host": None,
            "remote_port": None,
        }

        log.info("PULL[%s] P=%s D=%s", req_id[:8], p_url, d_url)

        resp = await self._forward_once(f"http://{p_url}/v1/chat/completions", p_body, headers)
        if resp.status_code != 200:
            return JSONResponse(status_code=resp.status_code, content={"error": resp.text[:200]})
        result = resp.json()

        token_ids = self._tokenize(body)
        self._record_blocks(idx, d_idx, token_ids, body.get("conversation_id"))

        kv_params = result.get("kv_transfer_params") or {}
        d_body = body.copy()
        d_body["kv_transfer_params"] = {
            "do_remote_decode": False,
            "do_remote_prefill": True,
            "remote_engine_id": kv_params.get("remote_engine_id") or eid,
            "remote_host": kv_params.get("remote_host") or self.p_kv_host,
            "remote_port": kv_params.get("remote_port") or sport,
            "remote_block_ids": kv_params.get("remote_block_ids"),
            "tp_size": kv_params.get("tp_size", self.p_tp_size),
            "pp_size": kv_params.get("pp_size", self.p_pp_size),
            "remote_request_id": req_id,
        }
        d_body.setdefault("max_tokens", body.get("max_tokens", 1024))

        try:
            d_stream = self._forward_stream(f"http://{d_url}/v1/chat/completions", d_body, headers)
        except RuntimeError as e:
            log.error("Decode-side failure for PULL[%s]: %s", req_id[:8], e)
            return JSONResponse(status_code=502, content={"error": str(e)})

        return StreamingResponse(d_stream, media_type="text/event-stream")


# ── FastAPI app ────────────────────────────────────────────────────

def create_app(**kwargs) -> FastAPI:
    app = FastAPI(title="Nano-Dynamo")
    proxy = DisaggProxy(**kwargs)
    app.state.proxy = proxy

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models():
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"http://{proxy.prefill_instances[0]}/v1/models")
            return JSONResponse(content=r.json())

    @app.post("/v1/chat/completions")
    async def chat_completions(raw_request: Request):
        if proxy.mode == "pull":
            return await proxy._pull(raw_request)
        return await proxy._push(raw_request)

    @app.post("/v1/completions")
    async def completions(raw_request: Request):
        if proxy.mode == "pull":
            return await proxy._pull(raw_request)
        return await proxy._push(raw_request)

    @app.get("/status")
    async def status():
        return {
            "mode": proxy.mode,
            "model": proxy.model,
            "prefill_instances": proxy.prefill_instances,
            "decode_instances": proxy.decode_instances,
            "policy": type(proxy.policy).__name__,
        }

    @app.get("/kvbm/status")
    async def kvbm_status():
        overloaded = proxy.migration.overloaded_workers()
        return {
            "decode_loads": {
                f"D{i}": round(proxy.migration.decode_load(i), 3)
                for i in range(len(proxy.decode_instances))
            },
            "overloaded": overloaded,
            "decode_usage": {
                f"D{i}": proxy.migration._decode_usage.get(i, 0)
                for i in range(len(proxy.decode_instances))
            },
        }

    @app.post("/kvbm/preempt")
    async def kvbm_preempt(raw_request: Request):
        body = await raw_request.json()
        d_idx = body.get("d_idx")
        count = body.get("count", proxy.preempt_batch)
        if d_idx is not None:
            evicted = proxy.preemptor.preempt(d_idx, count)
            return {"preempted": len(evicted), "sessions": [e.conv_id for e in evicted]}
        results = {}
        for didx in proxy.migration.overloaded_workers():
            evicted = proxy.preemptor.preempt(didx, count)
            results[f"D{didx}"] = len(evicted)
        return {"preempted": results}

    @app.get("/kvbm/preempted")
    async def kvbm_preempted():
        return {
            "count": proxy.preemptor.preempted_count(),
            "sessions": proxy.preemptor.preempted_list(),
        }

    @app.post("/scale/drain")
    async def scale_drain(raw_request: Request):
        body = await raw_request.json()
        role = body.get("role", "decode")
        idx = body.get("idx")
        if role == "prefill":
            result = proxy.scaler.drain_prefill(idx)
        else:
            result = proxy.scaler.drain_decode(idx)
        return result

    @app.post("/scale/activate")
    async def scale_activate(raw_request: Request):
        body = await raw_request.json()
        role = body.get("role", "decode")
        idx = body.get("idx")
        if role == "prefill":
            result = proxy.scaler.activate_prefill(idx)
        else:
            result = proxy.scaler.activate_decode(idx)
        return result

    @app.get("/scale/status")
    async def scale_status():
        return proxy.scaler.pool_metrics(proxy)

    @app.get("/scale/events")
    async def scale_events():
        return {"events": proxy.scaler.events()}

    # Phase 4+5 background tasks
    @app.on_event("startup")
    async def start_background_tasks():
        async def _preempt_loop():
            while True:
                await asyncio.sleep(5)
                for didx in proxy.migration.overloaded_workers():
                    load = proxy.migration.decode_load(didx)
                    if load >= proxy.preempt_threshold:
                        count = max(1, int((load - proxy.preempt_threshold) * 10))
                        evicted = proxy.preemptor.preempt(didx, count)
                        if evicted:
                            log.warning("PROACTIVE PREEMPT D%d load=%.2f evicted=%d",
                                        didx, load, len(evicted))
        async def _auto_scale_loop():
            while True:
                await asyncio.sleep(15)
                rec = proxy.scaler.auto_scale(proxy)
                if rec:
                    log.info("AUTOSCALE recommend: %s", rec)
        asyncio.create_task(_preempt_loop())
        asyncio.create_task(_auto_scale_loop())

    return app


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Nano-Dynamo Disaggregated Proxy")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--prefill-ports", type=int, nargs="+", default=[8100])
    parser.add_argument("--decode-ports", type=int, nargs="+", default=[8200])
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--mode", type=str, default="push", choices=["push", "pull"])
    parser.add_argument("--prefill-kv-host", type=str, default="127.0.0.1")
    parser.add_argument("--prefill-side-channel-ports", type=int, nargs="*", default=None)
    parser.add_argument("--prefill-tp-size", type=int, default=1)
    parser.add_argument("--prefill-pp-size", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--decode-cache-capacity", type=int, default=4096)
    parser.add_argument("--migration-threshold", type=float, default=0.80)
    parser.add_argument("--preempt-threshold", type=float, default=0.85)
    parser.add_argument("--preempt-batch", type=int, default=1)
    args = parser.parse_args()

    prefill_hosts = [f"127.0.0.1:{p}" for p in args.prefill_ports]
    decode_hosts = [f"127.0.0.1:{p}" for p in args.decode_ports]

    uvicorn.run(
        create_app(
            prefill_instances=prefill_hosts,
            decode_instances=decode_hosts,
            model=args.model,
            mode=args.mode,
            prefill_kv_host=args.prefill_kv_host,
            prefill_side_channel_ports=args.prefill_side_channel_ports,
            prefill_tp_size=args.prefill_tp_size,
            prefill_pp_size=args.prefill_pp_size,
            block_size=args.block_size,
            decode_cache_capacity=args.decode_cache_capacity,
            migration_threshold=args.migration_threshold,
            preempt_threshold=args.preempt_threshold,
            preempt_batch=args.preempt_batch,
        ),
        host=args.host,
        port=args.port,
    )
