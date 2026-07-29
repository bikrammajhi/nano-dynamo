from __future__ import annotations
import time
import xxhash
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .types import LocalBlockHash, ExternalSequenceBlockHash


def compute_hash(data: bytes) -> int:
    return xxhash.xxh3_64(data, seed=1337).intdigest()


def compute_block_hash_for_seq(token_ids: List[int], block_size: int) -> List[LocalBlockHash]:
    hashes: List[LocalBlockHash] = []
    for i in range(0, len(token_ids), block_size):
        chunk = token_ids[i : i + block_size]
        if len(chunk) < block_size:
            break
        raw = b"".join(t.to_bytes(4, "little") for t in chunk)
        hashes.append(LocalBlockHash(compute_hash(raw)))
    return hashes


@dataclass
class OverlapScores:
    scores: Dict[int, int] = field(default_factory=dict)

    def update_scores(self, workers: Set[int]) -> None:
        for w in workers:
            self.scores[w] = self.scores.get(w, 0) + 1


@dataclass
class KvCacheStoredBlockData:
    tokens_hash: LocalBlockHash
    block_hash: ExternalSequenceBlockHash


@dataclass
class KvCacheStoreData:
    parent_hash: Optional[ExternalSequenceBlockHash]
    blocks: List[KvCacheStoredBlockData]


@dataclass
class KvCacheEventData:
    Stored: KvCacheStoreData

    @staticmethod
    def stored(parent_hash: Optional[ExternalSequenceBlockHash], blocks: List[KvCacheStoredBlockData]) -> "KvCacheEventData":
        return KvCacheEventData(Stored=KvCacheStoreData(parent_hash=parent_hash, blocks=blocks))


@dataclass
class KvCacheEvent:
    event_id: int
    data: KvCacheEventData


@dataclass
class RouterEvent:
    worker_id: int
    event: KvCacheEvent


class RadixBlock:
    __slots__ = ("children", "workers", "recent_uses")

    def __init__(self) -> None:
        self.children: Dict[LocalBlockHash, RadixBlock] = {}
        self.workers: Set[int] = set()
        self.recent_uses: deque = deque()


class RadixTree:
    def __init__(self, expiration_duration: Optional[float] = None) -> None:
        self.root = RadixBlock()
        self.lookup: Dict[int, Dict[ExternalSequenceBlockHash, RadixBlock]] = {}
        self.expiration_duration = expiration_duration

    def find_matches(self, sequence: List[LocalBlockHash], early_exit: bool = False) -> OverlapScores:
        scores = OverlapScores()
        current: RadixBlock = self.root
        now = time.time()

        for block_hash in sequence:
            next_block = current.children.get(block_hash)
            if next_block is None:
                break
            scores.update_scores(next_block.workers)

            if self.expiration_duration is not None:
                while next_block.recent_uses and (now - next_block.recent_uses[0] > self.expiration_duration):
                    next_block.recent_uses.popleft()
                next_block.recent_uses.append(now)

            if early_exit and len(next_block.workers) == 1:
                break
            current = next_block

        return scores

    def apply_event(self, event: RouterEvent) -> None:
        worker_id = event.worker_id
        op = event.event.data
        worker_lookup = self.lookup.setdefault(worker_id, {})

        store = op.Stored
        current = worker_lookup.get(store.parent_hash) if store.parent_hash is not None else None
        if current is None:
            current = self.root

        for block_id in store.blocks:
            block = current.children.get(block_id.tokens_hash)
            if block is None:
                block = worker_lookup.get(block_id.block_hash) or RadixBlock()
                current.children[block_id.tokens_hash] = block
            block.workers.add(worker_id)
            worker_lookup[block_id.block_hash] = block
            current = block

    def remove_worker(self, worker_id: int) -> None:
        if worker_id not in self.lookup:
            return
        for block in self.lookup[worker_id].values():
            block.workers.discard(worker_id)
        del self.lookup[worker_id]


class KvIndexer:
    def __init__(self, block_size: int, expiration_duration: Optional[float] = None) -> None:
        self.tree = RadixTree(expiration_duration=expiration_duration)
        self.block_size = block_size

    def find_matches_for_request(self, token_ids: List[int]) -> OverlapScores:
        return self.tree.find_matches(compute_block_hash_for_seq(token_ids, self.block_size))

    def apply_event(self, event: RouterEvent) -> None:
        self.tree.apply_event(event)

    def remove_worker(self, worker_id: int) -> None:
        self.tree.remove_worker(worker_id)
