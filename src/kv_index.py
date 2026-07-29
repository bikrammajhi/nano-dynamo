"""KV-aware routing index — tracks which KV blocks each worker has cached.

Enables prefix-cache-aware request routing: route to the worker with highest
block overlap to eliminate redundant prefill computation.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set

log = logging.getLogger("kv-index")


@dataclass
class OverlapResult:
    scores: Dict[int, int] = field(default_factory=dict)


class KvIndex:
    def __init__(self):
        self._hash_to_workers: Dict[int, Set[int]] = defaultdict(set)

    def record_blocks(self, worker_index: int, block_hashes: List[int]):
        for h in block_hashes:
            self._hash_to_workers[h].add(worker_index)

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
