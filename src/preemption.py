"""Phase 4: Preemption & Promotion

Handles KV cache preemption (eviction) and session promotion
for disaggregated inference.

Preemption:
- When a decode worker is overloaded, the least recently used
  sessions are evicted to free KV cache blocks.
- Evicted blocks are removed from the block location registry
  so future routing won't assume they exist.

Promotion:
- When a preempted conversation returns, check if its blocks
  still exist on any decode worker. If yes, route back to that
  worker. If no, fall back to full re-prefill.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from .kvbm import MigrationManager

log = logging.getLogger("nano-dynamo.preempt")


@dataclass
class SessionInfo:
    conv_id: str
    block_hashes: Set[int]
    p_idx: int
    d_idx: int
    created_at: float
    last_access: float


@dataclass
class PreemptedRecord:
    conv_id: str
    block_hashes: Set[int]
    original_p_idx: int
    original_d_idx: int
    preempted_at: float


class PreemptionManager:
    """Tracks sessions, handles LRU eviction and promotion.

    Preemption at the gateway level means:
    - Mark a session's blocks as evicted from a decode worker
    - Release block location tracking so routing avoids the evicted worker
    - Track preempted sessions for potential promotion

    Promotion means:
    - When a preempted conversation returns, check if any decode
      worker still holds some of its blocks
    - If blocks survive, route to that worker (partial promotion)
    - Otherwise, full re-prefill via standard routing
    """

    def __init__(self, migration: MigrationManager):
        self.migration = migration

        # Active sessions: conv_id -> SessionInfo
        self._sessions: Dict[str, SessionInfo] = {}

        # Per-worker sessions: d_idx -> {conv_id -> SessionInfo}
        self._worker_sessions: Dict[int, Dict[str, SessionInfo]] = defaultdict(dict)

        # Preempted sessions: conv_id -> PreemptedRecord
        self._preempted: Dict[str, PreemptedRecord] = {}

    def register(
        self,
        conv_id: str,
        block_hashes: List[int],
        p_idx: int,
        d_idx: int,
    ):
        """Register an active session on a decode worker."""
        now = time.time()
        bh_set = set(block_hashes)
        info = SessionInfo(
            conv_id=conv_id, block_hashes=bh_set,
            p_idx=p_idx, d_idx=d_idx, created_at=now, last_access=now,
        )
        self._sessions[conv_id] = info
        self._worker_sessions[d_idx][conv_id] = info

    def access(self, conv_id: str):
        """Update last access time for LRU ordering."""
        if conv_id in self._sessions:
            self._sessions[conv_id].last_access = time.time()
        elif conv_id in self._preempted:
            self._preempted[conv_id].preempted_at = time.time()

    def preempt(self, d_idx: int, count: int = 1) -> List[PreemptedRecord]:
        """Preempt the N least-recently-used sessions from a decode worker.

        Removes blocks from the location registry and moves sessions
        to the preempted list for potential promotion.
        """
        sessions = sorted(
            self._worker_sessions.get(d_idx, {}).values(),
            key=lambda s: s.last_access,
        )
        evicted: List[PreemptedRecord] = []
        for session in sessions[:count]:
            now = time.time()
            record = PreemptedRecord(
                conv_id=session.conv_id,
                block_hashes=session.block_hashes,
                original_p_idx=session.p_idx,
                original_d_idx=session.d_idx,
                preempted_at=now,
            )
            self._preempted[session.conv_id] = record
            self.migration.registry.remove_worker_blocks(
                d_idx, "decode", session.block_hashes,
            )
            self.migration.release_decode_blocks(d_idx, list(session.block_hashes))
            del self._sessions[session.conv_id]
            del self._worker_sessions[d_idx][session.conv_id]
            evicted.append(record)
            log.info("PREEMPT conv=%s d_idx=%d blocks=%d age=%.1fs",
                     session.conv_id[:8], d_idx, len(session.block_hashes),
                     now - session.created_at)
        return evicted

    def promote(self, conv_id: str) -> Optional[PreemptedRecord]:
        """Attempt promotion: return record if conv_id was preempted.

        After promotion, the caller should re-route to the original
        decode worker (blocks may still survive from other recent
        requests) or let standard routing handle it.
        """
        record = self._preempted.pop(conv_id, None)
        if record is not None:
            log.info("PROMOTE conv=%s original_D=%d blocks=%d",
                     conv_id[:8], record.original_d_idx, len(record.block_hashes))
        return record

    def preempted_count(self) -> int:
        return len(self._preempted)

    def preempted_list(self) -> List[dict]:
        return [
            {
                "conv_id": r.conv_id[:8],
                "original_d_idx": r.original_d_idx,
                "block_count": len(r.block_hashes),
                "age_seconds": round(time.time() - r.preempted_at, 1),
            }
            for r in self._preempted.values()
        ]
