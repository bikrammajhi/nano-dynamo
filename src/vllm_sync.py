# SPDX-FileCopyrightText: Copyright (c) 2025 nano-dynamo contributors
# SPDX-License-Identifier: MIT

import asyncio
import json
import logging
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class VllmWorkerState:
    port: int
    gpu_id: int
    kv_role: str
    kv_cache_used_blocks: int = 0
    kv_cache_total_blocks: int = 0
    num_requests_running: int = 0
    num_requests_waiting: int = 0
    num_requests_swapped: int = 0
    last_sync_ts: float = 0.0
    prefix_cache_queries: int = 0
    prefix_cache_hits: int = 0
    prefix_cache_hit_rate: float = 0.0


class VllmBlockPoolSync:
    def __init__(self, workers: Dict[int, VllmWorkerState], total_blocks: int = 256, sync_interval: float = 0.5):
        self.workers = workers
        self.total_blocks = total_blocks
        self.sync_interval = sync_interval
        self._last_sync: float = 0.0
        for state in workers.values():
            state.kv_cache_total_blocks = total_blocks

    async def sync_now(self) -> Dict[int, VllmWorkerState]:
        now = time.time()
        if now - self._last_sync < self.sync_interval:
            return self.workers
        self._last_sync = now
        for wid, state in self.workers.items():
            try:
                await self._sync_worker(state)
            except Exception as e:
                logger.debug("sync failed for worker %d: %s", wid, e)
        return self.workers

    async def _sync_worker(self, state: VllmWorkerState):
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(None, self._fetch_metrics, state.port)
        if text is None:
            logger.debug("No metrics returned for worker port %d", state.port)
            return
        found_cache = False
        for line in text.split("\n"):
            if line.startswith("#") or not line.strip():
                continue
            # Split metric name from labels: "vllm:foo{labels} value" -> key="vllm:foo"
            # or "vllm:foo value" -> key="vllm:foo"
            if "{" in line:
                key = line.split("{")[0].strip()
                # Extract num_gpu_blocks from cache_config_info labels
                if key == "vllm:cache_config_info" and "num_gpu_blocks=" in line:
                    try:
                        labels_part = line.split("{")[1].split("}")[0]
                        for label in labels_part.split(","):
                            if label.startswith("num_gpu_blocks="):
                                total = int(label.split("=")[1].strip('"'))
                                if total > 0:
                                    self.total_blocks = total
                                    state.kv_cache_total_blocks = total
                    except (ValueError, IndexError):
                        pass
            else:
                key = line.split()[0] if line.split() else ""
            val = line.split()[-1] if line.split() else ""
            if key == "vllm:kv_cache_usage_perc":
                state.kv_cache_used_blocks = int(float(val) * self.total_blocks)
                found_cache = True
            elif key == "vllm:num_requests_running":
                state.num_requests_running = int(float(val))
            elif key == "vllm:num_requests_waiting":
                state.num_requests_waiting = int(float(val))
            elif key == "vllm:num_requests_swapped":
                state.num_requests_swapped = int(float(val))
            elif key in ("vllm:prefix_cache_queries_total", "vllm:prefix_cache_queries"):
                state.prefix_cache_queries = int(float(val))
            elif key in ("vllm:prefix_cache_hits_total", "vllm:prefix_cache_hits"):
                state.prefix_cache_hits = int(float(val))
        if state.prefix_cache_queries > 0:
            state.prefix_cache_hit_rate = state.prefix_cache_hits / state.prefix_cache_queries
        if not found_cache:
            logger.debug("No kv_cache_usage_perc found for port %d", state.port)

    def _check_alive(self, port: int) -> bool:
        try:
            req = urllib.request.Request(f"http://localhost:{port}/v1/models")
            resp = urllib.request.urlopen(req, timeout=2)
            return resp.status == 200
        except Exception:
            return False

    def _fetch_metrics(self, port: int) -> Optional[str]:
        try:
            req = urllib.request.Request(f"http://localhost:{port}/metrics")
            resp = urllib.request.urlopen(req, timeout=2)
            return resp.read().decode("utf-8")
        except Exception:
            return None

    def get_worker_loads(self) -> Dict[int, Dict]:
        return {
            wid: {
                "kv_cache_used_blocks": s.kv_cache_used_blocks,
                "kv_cache_total_blocks": self.total_blocks,
                "num_requests_running": s.num_requests_running,
                "num_requests_waiting": s.num_requests_waiting,
                "num_requests_swapped": s.num_requests_swapped,
                "prefix_cache_queries": s.prefix_cache_queries,
                "prefix_cache_hits": s.prefix_cache_hits,
                "prefix_cache_hit_rate": s.prefix_cache_hit_rate,
            }
            for wid, s in self.workers.items()
        }


@dataclass
class PrefillResult:
    request_id: str
    prefill_worker_id: int
    decode_worker_id: int
    num_tokens: int
    timestamp: float = 0.0


class VllmHttpOrchestrator:
    def __init__(self, workers: Dict[int, VllmWorkerState], timeout: float = 120.0):
        self.workers = workers
        self.timeout = timeout

    async def run_prefill(self, worker_id: int, model: str, prompt: str,
                          max_tokens: int = 1, temperature: float = 0.6,
                          top_p: float = 1.0) -> Optional[PrefillResult]:
        state = self.workers.get(worker_id)
        if state is None:
            logger.error("Prefill failed: worker %d not found", worker_id)
            return None
        payload = json.dumps({"model": model, "prompt": prompt, "max_tokens": max_tokens,
                              "temperature": temperature, "top_p": top_p}).encode()
        try:
            loop = asyncio.get_event_loop()
            resp_text = await loop.run_in_executor(None, self._http_post,
                                                   f"http://localhost:{state.port}/v1/completions", payload)
            resp = json.loads(resp_text)
            if "choices" not in resp:
                logger.warning("Prefill on worker %d returned no choices", worker_id)
                return None
            return PrefillResult(
                request_id=f"prefill_{worker_id}_{int(time.time()*1000)}",
                prefill_worker_id=worker_id, decode_worker_id=-1,
                num_tokens=len(prompt.split()), timestamp=time.time())
        except Exception as e:
            logger.error("Prefill failed on worker %d: %s", worker_id, e)
            return None

    async def run_decode(self, worker_id: int, model: str, prompt: str,
                         max_tokens: int = 128, temperature: float = 0.6,
                         top_p: float = 1.0, stream: bool = True) -> Optional[str]:
        state = self.workers.get(worker_id)
        if state is None:
            logger.error("Decode failed: worker %d not found", worker_id)
            return None
        payload = json.dumps({"model": model, "prompt": prompt, "max_tokens": max_tokens,
                              "temperature": temperature, "top_p": top_p, "stream": stream}).encode()
        try:
            loop = asyncio.get_event_loop()
            resp_text = await loop.run_in_executor(None, self._http_post,
                                                   f"http://localhost:{state.port}/v1/completions", payload)
            if stream:
                text = ""
                for line in resp_text.split("\n"):
                    if not line.startswith("data: "):
                        continue
                    chunk_str = line[6:].strip()
                    if chunk_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(chunk_str)
                        choices = chunk.get("choices", [])
                        if choices:
                            text += choices[0].get("text", "")
                    except json.JSONDecodeError:
                        continue
                return text
            resp = json.loads(resp_text)
            choices = resp.get("choices", [])
            return choices[0].get("text", "") if choices else None
        except Exception as e:
            logger.error("Decode failed on worker %d: %s", worker_id, e)
            return None

    async def transfer_kv(self, prefill_worker_id: int, decode_worker_id: int,
                          model: str, prompt: str, max_tokens: int = 128,
                          temperature: float = 0.6, top_p: float = 1.0
                          ) -> Tuple[Optional[str], Optional[PrefillResult]]:
        result = await self.run_prefill(worker_id=prefill_worker_id, model=model,
                                        prompt=prompt, max_tokens=1,
                                        temperature=temperature, top_p=top_p)
        if result is None:
            return None, None
        result.decode_worker_id = decode_worker_id
        text = await self.run_decode(worker_id=decode_worker_id, model=model, prompt=prompt,
                                     max_tokens=max_tokens, temperature=temperature, top_p=top_p)
        return text, result

    def _http_post(self, url: str, payload: bytes) -> str:
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=self.timeout)
        return resp.read().decode("utf-8")
