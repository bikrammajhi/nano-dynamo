"""Dashed zoom-in: KV Lookup. See docs/architecture.mmd (feeds S3/S6).

Single unified index (merge of the old KvIndex + BlockLocationRegistry, which
tracked identical hashes in parallel). One method answers both stages:
  S3 prefill selection -> overlap per prefill worker (role="prefill")
  S6 decode selection  -> overlap per decode worker  (role="decode")

Official counterpart: lib/kv-router/src/indexer/ + Router Design
`find_matches_for_request(tokens) -> {worker_id: matched_blocks}`.
Ours stores flat hash sets (not a radix tree) - same interface shape, a
fraction of the throughput. That tradeoff is the documented teaching cut.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple


class KvIndexer:
    """hash -> {(worker_id, role)}. Record ~= stored event, forget ~= removed."""

    def __init__(self):
        self._locations: Dict[int, Dict[Tuple[int, str], bool]] = {}

    def record(self, block_hashes: List[int], worker_id: int, role: str) -> None:
        for h in block_hashes:
            self._locations.setdefault(h, {})[(worker_id, role)] = True

    def forget(self, block_hashes: List[int], worker_id: int, role: str) -> None:
        """LRU eviction path (called by sessions.evict_lru)."""
        key = (worker_id, role)
        for h in block_hashes:
            holders = self._locations.get(h)
            if holders and key in holders:
                del holders[key]

    def worker_blocks(self, worker_id: int, role: str) -> Set[int]:
        key = (worker_id, role)
        return {h for h, holders in self._locations.items() if key in holders}

    def overlap(self, worker_id: int, role: str, hashes: Set[int]) -> int:
        return len(self.worker_blocks(worker_id, role) & hashes)

    def find_matches_for_request(
        self,
        token_ids: List[int],
        block_size: int,
        role: str = "prefill",
        worker_ids: Optional[List[int]] = None,
    ) -> Dict[int, int]:
        """Official-named entry: tokens in, per-worker match counts out.

        Hashing is delegated to infra/kv_events (partition + hash live there,
        lookup lives here - matches the Publisher/Indexer split). Workers
        outside worker_ids are not scored (eligibility already applied).
        """
        from infra.kv_events import token_ids_to_block_hashes

        request_hashes = set(token_ids_to_block_hashes(token_ids, block_size))
        if not request_hashes:
            return {}
        scores: Dict[int, int] = {}
        for h in request_hashes:
            for wid, r in self._locations.get(h, {}):
                if r == role and (worker_ids is None or wid in worker_ids):
                    scores[wid] = scores.get(wid, 0) + 1
        return scores
