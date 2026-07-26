from __future__ import annotations
import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .types import WorkerId, WorkerLoad
from .radix_tree import KvIndexer, OverlapScores


@dataclass(frozen=True)
class SchedulerConfig:
    overlap_score_credit: float = 1.0
    overlap_score_credit_decay: float = 0.0
    prefill_load_scale: float = 1.0
    host_cache_hit_weight: float = 0.75
    disk_cache_hit_weight: float = 0.25
    router_temperature: float = 0.0
    max_requests_per_worker: int = 16
    active_prefill_decay_factor: float = 0.9


@dataclass
class TierOverlapBlocks:
    device: Dict[int, int] = field(default_factory=dict)
    host_pinned: Dict[int, int] = field(default_factory=dict)
    disk: Dict[int, int] = field(default_factory=dict)


@dataclass
class OverlapSignals:
    tier_overlap_blocks: TierOverlapBlocks = field(default_factory=TierOverlapBlocks)
    effective_overlap_blocks: Dict[int, float] = field(default_factory=dict)
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

    def dequeue(self) -> Optional[QueuedRequest]:
        if self._queue:
            return self._queue.popleft()
        return None

    def __len__(self) -> int:
        return len(self._queue)

    def __bool__(self) -> bool:
        return bool(self._queue)


def _softmax_sample(logits: Dict[int, float], temperature: float) -> Tuple[int, float]:
    if not logits:
        raise ValueError("Empty logits for softmax sampling")
    if temperature == 0.0:
        best_worker = min(logits, key=lambda w: logits[w])
        return best_worker, logits[best_worker]

    workers = list(logits.keys())
    values = [logits[w] for w in workers]
    min_val, max_val = min(values), max(values)

    if min_val == max_val:
        idx = random.randint(0, len(workers) - 1)
        return workers[idx], values[idx]

    scale = -1.0 / ((max_val - min_val) * temperature)
    max_scaled = min_val * scale
    probs = [math.exp(v * scale - max_scaled) for v in values]
    total = sum(probs)
    probs = [p / total for p in probs]

    sample = random.random()
    cumsum = 0.0
    for i, prob in enumerate(probs):
        cumsum += prob
        if sample <= cumsum:
            return workers[i], values[i]
    return workers[-1], values[-1]


@dataclass
class KvScheduler:
    config: SchedulerConfig = field(default_factory=SchedulerConfig)
    indexer: KvIndexer | None = None
    block_size: int = 16
    _active_prefill_tokens: Dict[int, float] = field(default_factory=dict)
    _queue: RequestQueue = field(default_factory=RequestQueue)

    def _compute_worker_logit(
        self, isl_blocks: float, worker_load: WorkerLoad,
        overlap_signals: OverlapSignals, min_active_prefill_tokens: float
    ) -> float:
        wid = worker_load.worker_id.value
        tier = overlap_signals.tier_overlap_blocks
        device_blocks = tier.device.get(wid, 0)
        host_blocks = tier.host_pinned.get(wid, 0)
        disk_blocks = tier.disk.get(wid, 0)

        credit = (
            self.config.overlap_score_credit * device_blocks
            + self.config.host_cache_hit_weight * host_blocks
            + self.config.disk_cache_hit_weight * disk_blocks
        )

        if self.config.overlap_score_credit_decay > 0.0:
            excess_blocks = (worker_load.active_prefill_tokens - min_active_prefill_tokens) / self.block_size
            req_blocks = max(1, worker_load.kv_cache_total_blocks)
            credit /= 1.0 + self.config.overlap_score_credit_decay * (excess_blocks / req_blocks)

        cached_tokens = overlap_signals.effective_cached_tokens.get(wid, 0)
        raw_prefill_tokens = worker_load.active_prefill_tokens + max(
            0, (isl_blocks * self.block_size) - cached_tokens
        )
        adjusted = max(0.0, raw_prefill_tokens / self.block_size - credit)

        return self.config.prefill_load_scale * adjusted + float(worker_load.potential_decode_blocks())

    def _compute_overlap_credit(self, device_blocks: int = 0, host_blocks: int = 0, disk_blocks: int = 0) -> float:
        return (self.config.overlap_score_credit * device_blocks
                + self.config.host_cache_hit_weight * host_blocks
                + self.config.disk_cache_hit_weight * disk_blocks)

    def select_worker(
        self, token_ids: List[int], worker_loads: Dict[WorkerId, WorkerLoad]
    ) -> Optional[WorkerId]:
        if not worker_loads:
            raise ValueError("worker_loads must not be empty")

        overlap = (
            self.indexer.find_matches_for_request(token_ids)
            if self.indexer else OverlapScores()
        )

        overlap_signals = OverlapSignals()
        for wid_int, score in overlap.scores.items():
            overlap_signals.tier_overlap_blocks.device[wid_int] = score
            overlap_signals.effective_overlap_blocks[wid_int] = float(score)
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

        min_active = float("inf")
        if self.config.overlap_score_credit_decay > 0.0:
            min_active = min(load.active_prefill_tokens for _, load in candidates)
            if min_active == float("inf"):
                min_active = 0.0

        worker_logits = {
            wid.value: self._compute_worker_logit(isl_blocks, load, overlap_signals, min_active)
            for wid, load in candidates
        }

        selected_id, _ = _softmax_sample(worker_logits, self.config.router_temperature)

        decay = self.config.active_prefill_decay_factor
        current = self._active_prefill_tokens.get(selected_id, 0.0)
        self._active_prefill_tokens[selected_id] = decay * current + (1.0 - decay) * isl_tokens

        return WorkerId(selected_id)

    def update_worker_loads(self, worker_loads: Dict[WorkerId, WorkerLoad]) -> None:
        decay = self.config.active_prefill_decay_factor
        for wid, load in worker_loads.items():
            current = self._active_prefill_tokens.get(wid.value, 0.0)
            self._active_prefill_tokens[wid.value] = current * decay
            load.active_prefill_tokens = int(current * decay)

    def process_completed_request(self, worker_id: WorkerId, tokens_processed: int) -> None:
        current = self._active_prefill_tokens.get(worker_id.value, 0.0)
        self._active_prefill_tokens[worker_id.value] = max(0.0, current - tokens_processed)
