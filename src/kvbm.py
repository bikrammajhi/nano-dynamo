"""Phase 3: KV Block Migration (KVBM)

Tracks block placement across workers and coordinates migration
to balance KV cache pressure and optimize prefix reuse.

Migration strategy (pragmatic, no vLLM internals changes):
- Track which blocks each worker holds via block hashes
- When a decode worker approaches capacity, route future
  requests to less-loaded workers (eviction-based balancing)
- Migration is implemented as re-prefill on the optimal prefill
  worker, pushing directly to the target decode worker via NIXL
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

log = logging.getLogger("nano-dynamo.kvbm")


@dataclass
class BlockLocation:
    block_hash: int
    worker_id: int
    role: str
    last_seen: float = 0.0
    access_count: int = 0


class BlockLocationRegistry:
    """Tracks which KV block hashes live on which workers."""

    def __init__(self):
        self._locations: Dict[int, Dict[Tuple[int, str], BlockLocation]] = {}

    def record(self, block_hash: int, worker_id: int, role: str):
        now = time.time()
        key = (worker_id, role)
        if block_hash not in self._locations:
            self._locations[block_hash] = {}
        if key not in self._locations[block_hash]:
            self._locations[block_hash][key] = BlockLocation(
                block_hash=block_hash, worker_id=worker_id, role=role,
                last_seen=now, access_count=0,
            )
        loc = self._locations[block_hash][key]
        loc.last_seen = now
        loc.access_count += 1

    def worker_blocks(self, worker_id: int, role: str) -> Set[int]:
        blocks: Set[int] = set()
        for bh, locs in self._locations.items():
            if (worker_id, role) in locs:
                blocks.add(bh)
        return blocks

    def remove_worker_blocks(self, worker_id: int, role: str, block_hashes: Set[int]):
        for bh in block_hashes:
            key = (worker_id, role)
            if bh in self._locations and key in self._locations[bh]:
                del self._locations[bh][key]

    def overlap(self, worker_id: int, role: str, hashes: Set[int]) -> int:
        return len(self.worker_blocks(worker_id, role) & hashes)

    def overlap_decode(self, d_idx: int, hashes: Set[int]) -> int:
        return self.overlap(d_idx, "decode", hashes)


class MigrationManager:
    """Coordinates KV block tracking and migration-aware decode selection.

    Responsibilities:
    1. Track block placement across all workers
    2. Monitor decode worker cache pressure
    3. Select best decode worker based on overlap + load
    4. Plan migrations when imbalance is detected
    """

    def __init__(
        self,
        prefill_instances: List[str],
        decode_instances: List[str],
        decode_cache_capacity: int = 4096,
        migration_threshold: float = 0.80,
    ):
        self.registry = BlockLocationRegistry()
        self.prefill_instances = prefill_instances
        self.decode_instances = decode_instances
        self.decode_cache_capacity = decode_cache_capacity
        self.migration_threshold = migration_threshold

        self._decode_usage: Dict[int, int] = {}
        self._decode_blocks: Dict[int, Set[int]] = {}
        self._prefill_tokens: Dict[int, int] = {}
        self._prefill_requests: Dict[int, int] = {}

    def record_prefill_blocks(self, p_idx: int, block_hashes: List[int]):
        for bh in block_hashes:
            self.registry.record(bh, p_idx, "prefill")

    def record_prefill_load(self, p_idx: int, token_count: int):
        self._prefill_tokens[p_idx] = self._prefill_tokens.get(p_idx, 0) + token_count
        self._prefill_requests[p_idx] = self._prefill_requests.get(p_idx, 0) + 1

    def release_prefill_load(self, p_idx: int, token_count: int):
        current = self._prefill_tokens.get(p_idx, 0)
        self._prefill_tokens[p_idx] = max(0, current - token_count)
        reqs = self._prefill_requests.get(p_idx, 0)
        self._prefill_requests[p_idx] = max(0, reqs - 1)

    def active_prefill_tokens(self, p_idx: int) -> int:
        return self._prefill_tokens.get(p_idx, 0)

    def active_request_count(self, p_idx: int) -> int:
        return self._prefill_requests.get(p_idx, 0)

    def record_decode_blocks(self, d_idx: int, block_hashes: List[int]):
        existing = self._decode_blocks.get(d_idx, set())
        new_blocks = set(block_hashes) - existing
        self._decode_blocks[d_idx] = existing | new_blocks
        for bh in block_hashes:
            self.registry.record(bh, d_idx, "decode")
        self._decode_usage[d_idx] = self._decode_usage.get(d_idx, 0) + len(new_blocks)

    def release_decode_blocks(self, d_idx: int, block_hashes: List[int]):
        tracked = self._decode_blocks.get(d_idx, set())
        released = tracked & set(block_hashes)
        self._decode_blocks[d_idx] = tracked - released
        self._decode_usage[d_idx] = max(0, self._decode_usage.get(d_idx, 0) - len(released))

    def decode_load(self, d_idx: int) -> float:
        usage = self._decode_usage.get(d_idx, 0)
        return usage / max(self.decode_cache_capacity, 1)

    def potential_decode_blocks(self, d_idx: int) -> int:
        return self._decode_usage.get(d_idx, 0)

    def best_decode(self, token_hashes: Set[int], exclude: Optional[int] = None,
                    valid_only: Optional[List[int]] = None,
                    overlap_credit: Optional[float] = None) -> Tuple[int, int]:
        """Pick decode worker with lowest cost (Dynamo-equivalent).

        When overlap_credit is 0 (disaggregated decode mode), selection
        is purely load-based since KV is being transferred.

        Args:
            token_hashes: Set of block hashes to match.
            exclude: Worker index to exclude.
            valid_only: Only consider these worker indices (e.g. not draining).
            overlap_credit: Overlap credit multiplier (None=use config, 0=no credit).

        Returns (worker_idx, overlap_count).
        """
        candidates = valid_only if valid_only is not None else range(len(self.decode_instances))
        best_idx, best_logit, best_overlap = -1, float("inf"), -1
        tie_count = 0
        for d_idx in candidates:
            if d_idx == exclude:
                continue
            ov = self.registry.overlap_decode(d_idx, token_hashes)
            load = self._decode_usage.get(d_idx, 0)
            effective_credit = overlap_credit if overlap_credit is not None else 1.0
            adjusted_load = max(load - effective_credit * ov, 0)
            logit_val = adjusted_load
            if logit_val < best_logit:
                best_idx, best_logit, best_overlap = d_idx, logit_val, ov
                tie_count = 1
            elif logit_val == best_logit:
                tie_count += 1
                if random.randint(0, tie_count - 1) == 0:
                    best_idx, best_overlap = d_idx, ov
        if best_idx < 0:
            if valid_only:
                best_idx = min(valid_only, key=lambda i: self._decode_usage.get(i, 0))
            else:
                best_idx = min(range(len(self.decode_instances)),
                              key=lambda i: self._decode_usage.get(i, 0))
        return best_idx, best_overlap

    def overloaded_workers(self) -> List[int]:
        return [
            d_idx for d_idx in range(len(self.decode_instances))
            if self.decode_load(d_idx) >= self.migration_threshold
        ]
