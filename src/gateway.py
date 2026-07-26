#!/usr/bin/env python3
"""KV Router gateway — Dynamo-style routing for disaggregated prefill/decode.

Uses BlockPool + KvIndexer + KvScheduler from src/ for KV-aware routing,
with HTTP proxy to external vLLM instances.
"""
from __future__ import annotations

import asyncio
import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .types import WorkerId, WorkerLoad
from .radix_tree import (
    KvIndexer, RouterEvent, KvCacheEvent, KvCacheEventData,
    KvCacheStoreData, KvCacheStoredBlockData, ExternalSequenceBlockHash,
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


# ── Disaggregation decision (adapted from src/prefill.py) ─────────────

@dataclass
class DisaggregationDecision:
    should_prefill_remote: bool
    remote_worker_id: Optional[int]
    effective_prefill_length: int


def _decide_disagg(
    token_ids: List[int],
    local_worker_id: int,
    block_pools: Dict[int, BlockPool],
    worker_states: Dict[int, WorkerState],
    block_size: int,
    max_local_prefill_length: int = 512,
    queue_depth_threshold: int = 4,
) -> DisaggregationDecision:
    """Decide local vs remote prefill using BlockPool prefix matching."""
    total_tokens = len(token_ids)

    prefix_hit_tokens = 0
    pool = block_pools.get(local_worker_id)
    if pool:
        block_hashes = compute_block_hash_for_seq(token_ids, block_size)
        matched = pool.match_blocks(block_hashes)
        prefix_hit_tokens = matched.cached_count * block_size

    effective_prefill = max(0, total_tokens - prefix_hit_tokens)
    local_queue = worker_states[local_worker_id].active_requests

    if effective_prefill <= max_local_prefill_length and local_queue < queue_depth_threshold:
        return DisaggregationDecision(False, None, effective_prefill)

    # Pick least-loaded prefill worker as remote candidate
    candidates = sorted(
        [(wid, ws.active_requests) for wid, ws in worker_states.items() if wid != local_worker_id],
        key=lambda x: x[1],
    )
    if not candidates:
        return DisaggregationDecision(False, None, effective_prefill)

    return DisaggregationDecision(True, candidates[0][0], effective_prefill)


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
    ):
        self.prefill_urls = prefill_urls
        self.decode_urls = decode_urls
        self.block_size = block_size
        self.model_name = model_name

        self.prefill_states = {i: WorkerState(url) for i, url in enumerate(prefill_urls)}
        self.decode_states = {i: WorkerState(url) for i, url in enumerate(decode_urls)}

        # Dynamo components
        self.indexer = KvIndexer(block_size=block_size)
        self.scheduler = KvScheduler(
            config=SchedulerConfig(overlap_score_credit=1.0, router_temperature=0.0, max_requests_per_worker=16),
            indexer=self.indexer,
            block_size=block_size,
        )
        self.block_pools = {i: BlockPool(total_blocks=max_kv_blocks, block_size=block_size) for i in range(len(prefill_urls))}

        self.tokenizer = _get_tokenizer(model_name)

        self._req_counter = 0
        self._req_worker: Dict[str, int] = {}
        self._req_tokens: Dict[str, List[int]] = {}
        self._max_local_prefill = max_local_prefill_length

        log.info("Gateway: %d prefill + %d decode workers, block_size=%d", len(prefill_urls), len(decode_urls), block_size)

    def _get_worker_loads(self) -> Dict[WorkerId, WorkerLoad]:
        return {WorkerId(wid): ws.to_worker_load(wid) for wid, ws in self.prefill_states.items()}

    def _route(self, body: dict) -> tuple[str, str]:
        """Route request → (prefill_url, request_id)."""
        messages = body.get("messages", [])
        prompt = body.get("prompt", "")
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) if messages else prompt
        token_ids = self.tokenizer.encode(text) if text else []

        # 1. Cost function selection
        loads = self._get_worker_loads()
        worker_id = self.scheduler.select_worker(token_ids, loads)
        if worker_id is None:
            worker_id = WorkerId(self._req_counter % len(self.prefill_urls))
        wid = worker_id.value

        # 2. Disaggregation decision
        decision = _decide_disagg(
            token_ids, wid, self.block_pools, self.prefill_states,
            self.block_size, self._max_local_prefill,
        )
        if decision.should_prefill_remote and decision.remote_worker_id is not None:
            wid = decision.remote_worker_id

        self._req_counter += 1
        req_id = f"req-{self._req_counter}"
        self._req_worker[req_id] = wid
        self._req_tokens[req_id] = token_ids

        overlap = self.indexer.find_matches_for_request(token_ids)
        log.info("Route → prefill_%d (isl=%d, overlap=%d, remote=%s)",
                 wid, len(token_ids), overlap.scores.get(wid, 0), decision.should_prefill_remote)

        return self.prefill_urls[wid], req_id

    def _complete(self, request_id: str):
        """Commit blocks to BlockPool + update radix tree on request completion."""
        wid = self._req_worker.pop(request_id, None)
        token_ids = self._req_tokens.pop(request_id, None)
        if wid is None or token_ids is None:
            return

        block_hashes = compute_block_hash_for_seq(token_ids, self.block_size)
        if not block_hashes:
            return

        # Commit to BlockPool
        pool = self.block_pools[wid]
        try:
            block_ids = pool.allocate(len(block_hashes))
            for bid, bh in zip(block_ids, block_hashes):
                pool.commit(bid, bh)
        except RuntimeError:
            pass  # Pool full — skip, still update radix tree

        # Update radix tree
        blocks = [KvCacheStoredBlockData(tokens_hash=bh, block_hash=ExternalSequenceBlockHash(bh.value)) for bh in block_hashes]
        event = RouterEvent(
            worker_id=wid,
            event=KvCacheEvent(event_id=self._req_counter, data=KvCacheEventData.stored(parent_hash=None, blocks=blocks)),
        )
        self.indexer.apply_event(event)

    async def _update_loads(self):
        while True:
            await asyncio.gather(*(ws.scrape_metrics() for ws in self.prefill_states.values()))
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


def create_app(prefill_ports: List[int], decode_ports: List[int], **kwargs) -> FastAPI:
    app = FastAPI(title="Nano-Dynamo KV Router")
    prefill_urls = [f"http://127.0.0.1:{p}" for p in prefill_ports]
    decode_urls = [f"http://127.0.0.1:{p}" for p in decode_ports]

    app.state.gateway = KvRouterGateway(prefill_urls, decode_urls, **kwargs)

    @app.on_event("startup")
    async def startup():
        asyncio.create_task(app.state.gateway._update_loads())

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
    args = parser.parse_args()

    uvicorn.run(
        create_app(args.prefill_ports, args.decode_ports, model_name=args.model, max_local_prefill_length=args.max_local_prefill),
        host=args.host, port=args.port,
    )
