# SPDX-FileCopyrightText: Copyright (c) 2025 kv-prefix-router contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set

from .types import BlockId, LocalBlockHash


class KvBlockState(str, Enum):
    FREE = "free"
    INFLIGHT = "inflight"
    COMMITTED = "committed"


@dataclass
class KvBlock:
    block_id: BlockId
    state: KvBlockState
    block_hash: Optional[LocalBlockHash] = None


@dataclass(frozen=True)
class PrefillMatched:
    total_requested: int
    cached_count: int
    inflight_count: int

    @property
    def net_new_blocks(self) -> int:
        return self.total_requested - self.cached_count - self.inflight_count


class BlockPool:
    def __init__(self, total_blocks: int, block_size: int) -> None:
        if total_blocks <= 0:
            raise ValueError(f"total_blocks must be positive, got {total_blocks}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")

        self.total_blocks = total_blocks
        self.block_size = block_size
        self.blocks: List[Optional[KvBlock]] = [
            KvBlock(BlockId(i), KvBlockState.FREE) for i in range(total_blocks)
        ]
        self._free: deque = deque(range(total_blocks))
        self._committed_index: Dict[LocalBlockHash, BlockId] = {}
        self._eviction_queue: deque[LocalBlockHash] = deque()

    def _evict_until_free(self, needed: int) -> None:
        while len(self._free) < needed and self._eviction_queue:
            h = self._eviction_queue.popleft()
            bid = self._committed_index.pop(h, None)
            if bid is None:
                continue
            idx = bid.value
            block = self.blocks[idx]
            if block is not None and block.state == KvBlockState.COMMITTED:
                block.state = KvBlockState.FREE
                block.block_hash = None
                self._free.append(idx)

    def allocate(self, count: int) -> List[BlockId]:
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}")
        if count == 0:
            return []

        self._evict_until_free(count)

        if len(self._free) < count:
            raise RuntimeError(
                f"BlockPool: asked for {count} blocks, only {len(self._free)} free "
                f"(pool size {self.total_blocks})"
            )

        allocated: List[BlockId] = []
        for _ in range(count):
            idx = self._free.popleft()
            block = self.blocks[idx]
            assert block is not None
            block.state = KvBlockState.INFLIGHT
            block.block_hash = None
            allocated.append(block.block_id)

        return allocated

    def release(self, block_ids: List[BlockId]) -> None:
        for bid in block_ids:
            idx = bid.value
            block = self.blocks[idx]
            if block is None or block.state != KvBlockState.INFLIGHT:
                continue
            block.state = KvBlockState.FREE
            block.block_hash = None
            self._free.append(idx)

    def commit(self, block_id: BlockId, block_hash: LocalBlockHash) -> None:
        idx = block_id.value
        block = self.blocks[idx]
        if block is None:
            raise KeyError(f"BlockPool: commit failed, block {block_id} not found")
        if block.state != KvBlockState.INFLIGHT:
            raise RuntimeError(
                f"BlockPool: commit failed, block {block_id} is {block.state}, not INFLIGHT"
            )
        if block_hash in self._committed_index:
            old_bid = self._committed_index[block_hash]
            if old_bid.value != idx:
                old_block = self.blocks[old_bid.value]
                if old_block is not None and old_block.state == KvBlockState.COMMITTED:
                    old_block.state = KvBlockState.FREE
                    old_block.block_hash = None
                    self._free.append(old_bid.value)
        else:
            self._eviction_queue.append(block_hash)
        block.state = KvBlockState.COMMITTED
        block.block_hash = block_hash
        self._committed_index[block_hash] = block_id

    def match_blocks(self, requested_hashes: List[LocalBlockHash]) -> PrefillMatched:
        cached = 0
        inflight = 0
        inflight_hashes: Set[LocalBlockHash] = set()

        for block in self.blocks:
            if block is not None and block.state == KvBlockState.INFLIGHT:
                if block.block_hash is not None:
                    inflight_hashes.add(block.block_hash)

        for h in requested_hashes:
            if h in self._committed_index:
                cached += 1
            elif h in inflight_hashes:
                inflight += 1

        return PrefillMatched(
            total_requested=len(requested_hashes),
            cached_count=cached,
            inflight_count=inflight,
        )

    def __repr__(self) -> str:
        free = len(self._free)
        committed = len(self._committed_index)
        inflight = sum(1 for b in self.blocks if b is not None and b.state == KvBlockState.INFLIGHT)
        return (
            f"BlockPool(total={self.total_blocks}, "
            f"free={free}, committed={committed}, inflight={inflight})"
        )
