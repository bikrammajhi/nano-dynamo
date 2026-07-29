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

import asyncio
import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import httpx

log = logging.getLogger("nano-dynamo.kvbm")


@dataclass
class BlockLocation:
    block_hash: int
    worker_id: int
    role: str
    last_seen: float = 0.0
    access_count: int = 0


@dataclass
class MigrationPlan:
    request_id: str
    source_worker: int
    source_url: str
    dest_worker: int
    dest_url: str
    block_hashes: List[int]
    cost_estimate: float = 0.0


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

    def find(self, block_hash: int) -> List[BlockLocation]:
        return list(self._locations.get(block_hash, {}).values())

    def find_on_decode(self, block_hash: int) -> Optional[int]:
        for loc in self._locations.get(block_hash, {}).values():
            if loc.role == "decode":
                return loc.worker_id
        return None

    def find_on_prefill(self, block_hash: int) -> Optional[int]:
        for loc in self._locations.get(block_hash, {}).values():
            if loc.role == "prefill":
                return loc.worker_id
        return None

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
        self._active_migrations: Dict[str, asyncio.Task] = {}
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))
        return self._client

    def record_prefill_blocks(self, p_idx: int, block_hashes: List[int]):
        for bh in block_hashes:
            self.registry.record(bh, p_idx, "prefill")

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

    def best_decode(self, token_hashes: Set[int], exclude: Optional[int] = None,
                    valid_only: Optional[List[int]] = None) -> Tuple[int, int]:
        """Pick decode worker with most prefix overlap, breaking ties by load.

        Args:
            token_hashes: Set of block hashes to match.
            exclude: Worker index to exclude.
            valid_only: Only consider these worker indices (e.g. not draining).

        Returns (worker_idx, overlap_count).
        """
        candidates = valid_only if valid_only is not None else range(len(self.decode_instances))
        best_idx, best_overlap, best_load = -1, -1, float("inf")
        for d_idx in candidates:
            if d_idx == exclude:
                continue
            ov = self.registry.overlap_decode(d_idx, token_hashes)
            load = self._decode_usage.get(d_idx, 0)
            if ov > best_overlap or (ov == best_overlap and load < best_load):
                best_idx, best_overlap, best_load = d_idx, ov, load
            elif ov == best_overlap and load == best_load and best_idx >= 0:
                if random.random() < 0.5:
                    best_idx, best_overlap, best_load = d_idx, ov, load
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

    def plan_migration(
        self,
        target_decode: int,
        required_hashes: Set[int],
    ) -> Optional[MigrationPlan]:
        existing = self.registry.worker_blocks(target_decode, "decode")
        missing = required_hashes - existing
        if not missing:
            return None

        best_p, best_ov = -1, -1
        for p_idx in range(len(self.prefill_instances)):
            p_blocks = self.registry.worker_blocks(p_idx, "prefill")
            ov = len(missing & p_blocks)
            if ov > best_ov:
                best_ov = ov
                best_p = p_idx

        if best_p < 0 or best_ov == 0:
            return None

        return MigrationPlan(
            request_id=str(uuid.uuid4()),
            source_worker=best_p,
            source_url=self.prefill_instances[best_p],
            dest_worker=target_decode,
            dest_url=self.decode_instances[target_decode],
            block_hashes=list(missing & self.registry.worker_blocks(best_p, "prefill")),
            cost_estimate=best_ov,
        )

    async def close(self):
        if self._client:
            await self._client.aclose()
