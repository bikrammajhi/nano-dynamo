#!/usr/bin/env python3
"""KV Router gateway — Dynamo-style routing for disaggregated prefill/decode.

Uses BlockPool + KvIndexer + KvScheduler from src/ for KV-aware routing,
with HTTP proxy to external vLLM instances.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import zmq
import zmq.asyncio

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .prefill import DisaggregatedRouter
from .types import LocalBlockHash, WorkerId, WorkerLoad
from .radix_tree import (
    KvIndexer, RouterEvent, KvCacheEvent, KvCacheEventData,
    KvCacheStoredBlockData, ExternalSequenceBlockHash,
    compute_block_hash_for_seq,
)
from .scheduler import KvScheduler, SchedulerConfig
from .block_pool import BlockPool

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("kv-router")

_tokenizer = None
_tokenizer_lock = threading.Lock()


def _get_tokenizer(model_name: str):
    global _tokenizer
    if _tokenizer is None:
        with _tokenizer_lock:
            if _tokenizer is None:
                from transformers import AutoTokenizer
                log.info("Loading tokenizer for %s...", model_name)
                _tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return _tokenizer


# ── Worker state (metrics scraped from vLLM /metrics) ─────────────────

@dataclass
class WorkerState:
    url: str
    active_requests: int = 0
    kv_cache_used_blocks: int = 0
    kv_cache_total_blocks: int = 100

    def to_worker_load(self, worker_id: int) -> WorkerLoad:
        return WorkerLoad(
            worker_id=WorkerId(worker_id),
            active_requests=self.active_requests,
            kv_cache_used_blocks=self.kv_cache_used_blocks,
            kv_cache_total_blocks=self.kv_cache_total_blocks,
        )

    async def scrape_metrics(self):
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{self.url}/metrics")
                if resp.status_code != 200:
                    return
                for line in resp.read().decode().split("\n"):
                    if line.startswith("#") or not line.strip() or "{" in line:
                        continue
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    if "num_requests_running" in line:
                        self.active_requests = int(float(parts[1]))
                    elif "gpu_cache_usage_perc" in line:
                        self.kv_cache_used_blocks = int(float(parts[1]) * self.kv_cache_total_blocks)
        except Exception:
            pass


# ── Core gateway ──────────────────────────────────────────────────────

class KvRouterGateway:
    def __init__(
        self,
        prefill_urls: List[str],
        decode_urls: List[str],
        model_name: str,
        block_size: int = 64,
        max_kv_blocks: int = 100,
        max_local_prefill_length: int = 512,
        kv_events_ports: Optional[List[int]] = None,
    ):
        self.prefill_urls = prefill_urls
        self.decode_urls = decode_urls
        self.block_size = block_size
        self.model_name = model_name
        self.kv_events_ports = kv_events_ports or []

        self.prefill_states = {i: WorkerState(url) for i, url in enumerate(prefill_urls)}
        self.decode_states = {i: WorkerState(url) for i, url in enumerate(decode_urls)}

        # Dynamo components
        self.indexer = KvIndexer(block_size=block_size)
        self.scheduler = KvScheduler(
            config=SchedulerConfig(overlap_weight=2.0, max_requests_per_worker=16),
            indexer=self.indexer,
            block_size=block_size,
        )
        worker_ids = [WorkerId(i) for i in range(len(prefill_urls))]
        self.block_pools = {i: BlockPool(total_blocks=max_kv_blocks, block_size=block_size) for i in range(len(prefill_urls))}
        self.disagg_router = DisaggregatedRouter(
            worker_ids=worker_ids,
            block_pools={wid: self.block_pools[wid.value] for wid in worker_ids},
            block_size=block_size,
            max_local_prefill_length=max_local_prefill_length,
        )

        self.tokenizer = _get_tokenizer(model_name)

        self._req_counter = 0
        self._req_worker: Dict[str, int] = {}
        self._req_block_hashes: Dict[str, List[LocalBlockHash]] = {}
        self._req_decode: Dict[str, Optional[int]] = {}

        log.info("Gateway: %d prefill + %d decode workers, block_size=%d, kv_events=%s",
                 len(prefill_urls), len(decode_urls), block_size, bool(self.kv_events_ports))

    def _get_worker_loads(self) -> Dict[WorkerId, WorkerLoad]:
        return {WorkerId(wid): ws.to_worker_load(wid) for wid, ws in self.prefill_states.items()}

    def _get_decode_loads(self) -> Dict[int, float]:
        if not self.decode_states:
            return {}
        max_active = max((s.active_requests for s in self.decode_states.values()), default=1)
        return {did: s.active_requests / max(max_active, 1) for did, s in self.decode_states.items()}

    def _route(self, body: dict) -> tuple[str, str]:
        """Route request → (prefill_url, request_id).

        Selects prefill worker via cost function (overlap + cache + decode load),
        then optionally offloads via disaggregated router.
        Update the radix tree optimistically at routing time.
        """
        messages = body.get("messages", [])
        prompt = body.get("prompt", "")
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) if messages else prompt
        token_ids = self.tokenizer.encode(text) if text else []

        # 1. Cost function selection (prefill load + decode load)
        loads = self._get_worker_loads()
        decode_loads = self._get_decode_loads()
        worker_id = self.scheduler.select_worker(token_ids, loads, decode_loads=decode_loads)
        if worker_id is None:
            worker_id = WorkerId(self._req_counter % len(self.prefill_urls))
        wid = worker_id.value

        # 2. Disaggregation decision
        worker_loads = {WorkerId(k): v.active_requests for k, v in self.prefill_states.items()}
        decision = self.disagg_router.should_prefill_remote(
            token_ids, WorkerId(wid), worker_loads,
        )
        if decision.should_prefill_remote and decision.remote_worker_id is not None:
            wid = decision.remote_worker_id.value

        # 3. Pick decode worker (paired by index)
        decode_id = wid % len(self.decode_states) if self.decode_states else None

        self._req_counter += 1
        req_id = f"req-{self._req_counter}"
        self._req_worker[req_id] = wid
        self._req_decode[req_id] = decode_id

        # 4. Eagerly update the radix tree — blocks are being computed NOW on wid
        block_hashes = compute_block_hash_for_seq(token_ids, self.block_size)
        if block_hashes:
            blocks = [
                KvCacheStoredBlockData(tokens_hash=bh, block_hash=ExternalSequenceBlockHash(bh.value))
                for bh in block_hashes
            ]
            event = RouterEvent(
                worker_id=wid,
                event=KvCacheEvent(
                    event_id=self._req_counter,
                    data=KvCacheEventData.stored(parent_hash=None, blocks=blocks),
                ),
            )
            self.indexer.apply_event(event)
            self._req_block_hashes[req_id] = block_hashes

        overlap = self.indexer.find_matches_for_request(token_ids)
        decode_usage = decode_loads.get(wid, 0.0) if decode_id is not None else 0.0
        log.info("Route → prefill_%d decode_%s (isl=%d, overlap=%d, decode_usage=%.2f, remote=%s)",
                 wid, decode_id, len(token_ids), overlap.scores.get(wid, 0),
                 decode_usage, decision.should_prefill_remote)

        return self.prefill_urls[wid], req_id

    def _complete(self, request_id: str):
        """Commit blocks to BlockPool on request completion (evicts if full)."""
        wid = self._req_worker.pop(request_id, None)
        if wid is None:
            return

        block_hashes = self._req_block_hashes.pop(request_id, None)
        if not block_hashes:
            return

        pool = self.block_pools[wid]
        block_ids = pool.allocate(len(block_hashes))
        for bid, bh in zip(block_ids, block_hashes):
            pool.commit(bid, bh)

    async def _kv_events_listener(self, worker_id: int, port: int):
        """Subscribe to vLLM KV events via ZMQ and update the radix tree in real-time."""
        ctx = zmq.asyncio.Context()
        socket = ctx.socket(zmq.SUB)
        socket.connect(f"tcp://127.0.0.1:{port}")
        socket.subscribe("kv-events")
        log.info("ZMQ subscriber for worker_%d connected to port %d", worker_id, port)
        while True:
            try:
                _, payload = await socket.recv_multipart()
                msg: dict[str, Any] = json.loads(payload)
                event_id = msg.get("event_id", 0)
                etype = msg.get("type", "")
                blocks_raw = msg.get("blocks", [])

                if etype == "stored":
                    blocks = [
                        KvCacheStoredBlockData(
                            tokens_hash=LocalBlockHash(b["token_hash"]),
                            block_hash=ExternalSequenceBlockHash(b["block_hash"]),
                        )
                        for b in blocks_raw if "token_hash" in b and "block_hash" in b
                    ]
                    if not blocks:
                        continue
                    parent_hash_raw = msg.get("parent_hash")
                    parent_hash = ExternalSequenceBlockHash(parent_hash_raw) if parent_hash_raw is not None else None
                    event = RouterEvent(
                        worker_id=worker_id,
                        event=KvCacheEvent(
                            event_id=event_id,
                            data=KvCacheEventData.stored(parent_hash=parent_hash, blocks=blocks),
                        ),
                    )
                    self.indexer.apply_event(event)
                    log.debug("ZMQ stored: worker_%d %d blocks (event=%d)", worker_id, len(blocks), event_id)

                elif etype == "removed":
                    block_hashes = [
                        ExternalSequenceBlockHash(b["block_hash"])
                        for b in blocks_raw if "block_hash" in b
                    ]
                    if not block_hashes:
                        continue
                    event = RouterEvent(
                        worker_id=worker_id,
                        event=KvCacheEvent(
                            event_id=event_id,
                            data=KvCacheEventData.removed(block_hashes=block_hashes),
                        ),
                    )
                    self.indexer.apply_event(event)
                    log.debug("ZMQ removed: worker_%d %d blocks (event=%d)", worker_id, len(block_hashes), event_id)

            except Exception:
                log.exception("ZMQ subscriber error for worker_%d", worker_id)
                await asyncio.sleep(1)

    async def _update_loads(self):
        while True:
            all_states = list(self.prefill_states.values()) + list(self.decode_states.values())
            await asyncio.gather(*(ws.scrape_metrics() for ws in all_states))
            self.scheduler.update_worker_loads(self._get_worker_loads())
            await asyncio.sleep(2)


# ── FastAPI app ───────────────────────────────────────────────────────

def _make_proxy_handler(endpoint: str):
    """Create a streaming + non-streaming proxy handler for an endpoint."""
    async def handler(request: Request):
        body = await request.json()
        is_stream = body.get("stream", False)
        gw: KvRouterGateway = request.app.state.gateway

        prefill_url, req_id = gw._route(body)
        target = f"{prefill_url}{endpoint}"

        if is_stream:
            async def stream():
                try:
                    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
                        async with client.stream("POST", target, json=body, headers={"Content-Type": "application/json"}) as resp:
                            if resp.status_code != 200:
                                err = b""
                                async for chunk in resp.aiter_bytes():
                                    err += chunk
                                log.error("Upstream %d: %s", resp.status_code, err[:200])
                                yield f"data: {err.decode()}\n\n"
                                return
                            async for chunk in resp.aiter_bytes():
                                yield chunk
                finally:
                    gw._complete(req_id)
            return StreamingResponse(stream(), media_type="text/event-stream")
        else:
            async with httpx.AsyncClient(timeout=300.0) as client:
                resp = await client.post(target, json=body, headers={"Content-Type": "application/json"})
            gw._complete(req_id)
            if resp.status_code != 200:
                log.error("Upstream %d: %s", resp.status_code, resp.text[:200])
                return JSONResponse(status_code=resp.status_code, content={"error": resp.text[:200]})
            return JSONResponse(content=resp.json())

    return handler


def create_app(prefill_ports: List[int], decode_ports: List[int], kv_events_ports: Optional[List[int]] = None, **kwargs) -> FastAPI:
    app = FastAPI(title="Nano-Dynamo KV Router")
    prefill_urls = [f"http://127.0.0.1:{p}" for p in prefill_ports]
    decode_urls = [f"http://127.0.0.1:{p}" for p in decode_ports]

    app.state.gateway = KvRouterGateway(prefill_urls, decode_urls, kv_events_ports=kv_events_ports, **kwargs)

    @app.on_event("startup")
    async def startup():
        gw = app.state.gateway
        asyncio.create_task(gw._update_loads())
        for i, port in enumerate(gw.kv_events_ports):
            asyncio.create_task(gw._kv_events_listener(i, port))

    @app.get("/v1/models")
    async def list_models():
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{prefill_urls[0]}/v1/models")
            return JSONResponse(content=resp.json())

    app.post("/v1/chat/completions")(_make_proxy_handler("/v1/chat/completions"))
    app.post("/v1/completions")(_make_proxy_handler("/v1/completions"))
    app.get("/health")(lambda: {"status": "ok"})

    return app


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Nano-Dynamo KV Router")
    parser.add_argument("--prefill-ports", type=int, nargs="+", default=[8100])
    parser.add_argument("--decode-ports", type=int, nargs="+", default=[8200])
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--max-local-prefill", type=int, default=512)
    parser.add_argument("--kv-events-ports", type=int, nargs="*", default=None,
                        help="ZMQ ports for vLLM KV events (one per prefill worker)")
    args = parser.parse_args()

    uvicorn.run(
        create_app(args.prefill_ports, args.decode_ports, kv_events_ports=args.kv_events_ports,
                   model_name=args.model, max_local_prefill_length=args.max_local_prefill),
        host=args.host, port=args.port,
    )
