from __future__ import annotations
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .types import WorkerId, WorkerLoad
from .radix_tree import KvIndexer, OverlapScores


@dataclass(frozen=True)
class SchedulerConfig:
    overlap_weight: float = 2.0
    decode_usage_weight: float = 1.0
    max_requests_per_worker: int = 16


@dataclass
class OverlapSignals:
    overlap_blocks: Dict[int, int] = field(default_factory=dict)
    effective_cached_tokens: Dict[int, int] = field(default_factory=dict)


@dataclass
class QueuedRequest:
    token_ids: List[int]
    worker_loads: Dict[WorkerId, WorkerLoad]
    enqueue_time: float = field(default_factory=lambda: __import__("time").time())


class RequestQueue:
    def __init__(self, max_size: int = 1000) -> None:
        self._queue: deque[QueuedRequest] = deque(maxlen=max_size)

    def enqueue(self, request: QueuedRequest) -> bool:
        if len(self._queue) >= self._queue.maxlen:
            return False
        self._queue.append(request)
        return True

    def __len__(self) -> int:
        return len(self._queue)

    def __bool__(self) -> bool:
        return bool(self._queue)


@dataclass
class KvScheduler:
    config: SchedulerConfig = field(default_factory=SchedulerConfig)
    indexer: KvIndexer | None = None
    block_size: int = 16
    _queue: RequestQueue = field(default_factory=RequestQueue)

    def _compute_worker_logit(
        self, isl_blocks: float, worker_load: WorkerLoad,
        overlap_signals: OverlapSignals, max_waiting: int,
        decode_usage: float = 0.0,
    ) -> float:
        wid = worker_load.worker_id.value
        matching_blocks = overlap_signals.overlap_blocks.get(wid, 0)
        score = matching_blocks / isl_blocks if isl_blocks > 0 else 0.0

        total_blocks = worker_load.kv_cache_total_blocks
        cache_usage = (
            worker_load.kv_cache_used_blocks / total_blocks
            if total_blocks > 0 else 0.0
        )

        waiting = (
            worker_load.active_requests / max_waiting
            if max_waiting > 0 else 0.0
        )

        return self.config.overlap_weight * score - cache_usage - waiting - self.config.decode_usage_weight * decode_usage

    def select_worker(
        self, token_ids: List[int], worker_loads: Dict[WorkerId, WorkerLoad],
        decode_loads: Optional[Dict[int, float]] = None,
    ) -> Optional[WorkerId]:
        if not worker_loads:
            raise ValueError("worker_loads must not be empty")

        overlap = (
            self.indexer.find_matches_for_request(token_ids)
            if self.indexer else OverlapScores()
        )

        overlap_signals = OverlapSignals()
        for wid_int, score in overlap.scores.items():
            overlap_signals.overlap_blocks[wid_int] = score
            overlap_signals.effective_cached_tokens[wid_int] = score * self.block_size

        isl_tokens = len(token_ids)
        isl_blocks = isl_tokens / self.block_size

        if isl_tokens == 0:
            for wid, load in worker_loads.items():
                if load.active_requests < self.config.max_requests_per_worker:
                    return wid
            return None

        candidates = [
            (wid, load) for wid, load in worker_loads.items()
            if load.active_requests < self.config.max_requests_per_worker
            and load.kv_cache_used_blocks < load.kv_cache_total_blocks
        ]

        if not candidates:
            self._queue.enqueue(QueuedRequest(token_ids=token_ids, worker_loads=worker_loads))
            return None

        max_waiting = max(load.active_requests for _, load in candidates)

        worker_logits = {
            wid.value: self._compute_worker_logit(
                isl_blocks, load, overlap_signals, max_waiting,
                decode_usage=(decode_loads or {}).get(wid.value, 0.0),
            )
            for wid, load in candidates
        }

        max_logit = max(worker_logits.values())
        best_workers = [wid for wid, logit in worker_logits.items() if logit == max_logit]
        selected_id = random.choice(best_workers)

        return WorkerId(selected_id)

    def update_worker_loads(self, worker_loads: Dict[WorkerId, WorkerLoad]) -> None:
        pass
