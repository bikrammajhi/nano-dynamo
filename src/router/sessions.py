"""Dashed note on S3 entry: session affinity (hard mode). See docs/architecture.mmd.

Official counterparts: Tuning doc --router-session-affinity-* (+ X-Dynamo-Session-ID
header; ours is the `conversation_id` body field) and
lib/kv-router/src/session_prefix_index.rs. Hard mode semantics, verbatim:
exact-dispatch to the stored target; invalid target -> invalidate + normal
selection once. No TTL expiry here (documented gap: official idle-TTL has no
nano equivalent; bindings live until invalidated or evicted).

Eviction doubles as the official "KV removed" path: evict_lru drops the
session's decode blocks from indexer + slots, so future S3/S6 scoring stops
seeing them - same observable effect as engine-published removed events.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from router.kv_indexer import KvIndexer
from router.slot_manager import SlotManager


@dataclass
class Binding:
    """One session's placement + the blocks justifying it."""
    route: object  # router.prefill_router.Route (duck-typed: no router import)
    block_hashes: Set[int] = field(default_factory=set)
    last_touch: float = field(default_factory=time.monotonic)


class SessionAffinity:
    """Router-side affinity coordinator. Bindings are advisory local state
    (official: never authoritative storage), owned here and only here."""

    def __init__(self, indexer: KvIndexer, slots: SlotManager):
        self.indexer = indexer
        self.slots = slots
        self._bindings: Dict[str, Binding] = {}

    def lookup(self, session_id: str):
        """Return bound Route or None. Hit refreshes the LRU clock."""
        b = self._bindings.get(session_id)
        if b is None:
            return None
        b.last_touch = time.monotonic()
        return b.route

    def bind(self, session_id: str, route, block_hashes: Set[int]) -> None:
        """Commit AFTER successful dispatch (official: commit-on-dispatch, so a
        failed route never pins a session to a dead worker)."""
        self._bindings[session_id] = Binding(
            route=route, block_hashes=set(block_hashes),
            last_touch=time.monotonic(),
        )

    def invalidate(self, session_id: str) -> None:
        """Drop the binding; next request re-selects normally (hard-mode rule)."""
        self._bindings.pop(session_id, None)

    def access(self, session_id: str) -> None:
        """LRU touch for sessions seen but routed normally."""
        b = self._bindings.get(session_id)
        if b is not None:
            b.last_touch = time.monotonic()

    def evict_lru(self, d_idx: int, count: int = 1) -> List[str]:
        """Evict the N least-recently-used sessions on decode worker d_idx:
        forget their decode blocks (indexer) + release usage (slots), drop
        bindings. Returns evicted session ids, oldest first."""
        cands = sorted(
            ((sid, b) for sid, b in self._bindings.items()
             if getattr(b.route, "d_idx", None) == d_idx),
            key=lambda kv: kv[1].last_touch,
        )
        evicted = []
        for sid, b in cands[:count]:
            self.indexer.forget(list(b.block_hashes), d_idx, "decode")
            self.slots.free(getattr(b.route, "p_idx", 0), d_idx, len(b.block_hashes))
            del self._bindings[sid]
            evicted.append(sid)
        return evicted
