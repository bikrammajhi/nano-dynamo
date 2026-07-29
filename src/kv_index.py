"""KV-aware routing index — tracks which KV blocks each worker has cached.

Enables prefix-cache-aware request routing: route to the worker with highest
block overlap to eliminate redundant prefill computation.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

log = logging.getLogger("kv-index")


@dataclass
class OverlapResult:
    scores: Dict[int, int] = field(default_factory=dict)
    """worker_index → number of overlapping blocks"""

    def best_worker(self) -> Optional[int]:
        if not self.scores:
            return None
        return max(self.scores, key=self.scores.get)


class KvIndex:
    """Tracks KV block hashes per worker for prefix-aware routing.

    Uses a simple hash-table approach: maps each block hash to the set
    of workers that have it cached. Prefix overlap is computed by
    counting common hashes between a request and each worker's cache.
    """

    def __init__(self):
        self._hash_to_workers: Dict[int, Set[int]] = defaultdict(set)
        self._worker_hashes: Dict[int, Set[int]] = defaultdict(set)

    def record_blocks(self, worker_index: int, block_hashes: List[int]):
        for h in block_hashes:
            self._hash_to_workers[h].add(worker_index)
            self._worker_hashes[worker_index].add(h)

    def remove_worker(self, worker_index: int):
        hashes = self._worker_hashes.pop(worker_index, set())
        for h in hashes:
            self._hash_to_workers[h].discard(worker_index)
            if not self._hash_to_workers[h]:
                del self._hash_to_workers[h]

    def compute_overlap(self, token_ids: List[int], block_size: int) -> OverlapResult:
        from .kv_events import token_ids_to_block_hashes
        request_hashes = set(token_ids_to_block_hashes(token_ids, block_size))
        if not request_hashes:
            return OverlapResult()
        scores: Dict[int, int] = {}
        for h in request_hashes:
            for w in self._hash_to_workers.get(h, set()):
                scores[w] = scores.get(w, 0) + 1
        return OverlapResult(scores=scores)

    def worker_cache_size(self, worker_index: int) -> int:
        return len(self._worker_hashes.get(worker_index, set()))
