"""Dashed zoom-in: Load Snapshot. See docs/architecture.mmd (feeds S3/S6).

Ephemeral active-load counters - the "same worker-load snapshot" the official
cost model reads. Nothing here is durable: restart wipes it (like the indexer).
Official counterparts: lib/kv-router/src/sequences/ (active-sequence tracking)
and the replica-sync lifecycle AddRequest / MarkPrefillCompleted / Free, whose
names the methods below carry verbatim.

Lifecycle (mirrors disaggregated request flow):
  add_request           (dispatch: charge full ISL to prefill + blocks to decode)
  mark_prefill_completed(first token: release prompt-side prefill charge only)
  free                  (finish: release decode blocks)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class WorkerLoad:
    """One worker's snapshot row. Owned here (it is snapshot()'s return)."""
    active_prefill_tokens: int = 0
    active_decode_blocks: int = 0
    active_requests: int = 0


class SlotManager:
    def __init__(self):
        self._prefill_tokens: Dict[int, int] = {}
        self._prefill_requests: Dict[int, int] = {}
        self._decode_blocks: Dict[int, int] = {}

    def add_request(self, p_idx: int, d_idx: int,
                    isl_tokens: int, block_count: int) -> None:
        self._prefill_tokens[p_idx] = self._prefill_tokens.get(p_idx, 0) + isl_tokens
        self._prefill_requests[p_idx] = self._prefill_requests.get(p_idx, 0) + 1
        self._decode_blocks[d_idx] = self._decode_blocks.get(d_idx, 0) + block_count

    def mark_prefill_completed(self, p_idx: int, isl_tokens: int) -> None:
        """First token: prompt work is done, decode continues. Release ONLY the
        prefill side - decode blocks stay charged until free(). (Old code
        released at full-drain completion; first-token timing matches the
        official MarkPrefillCompleted semantic.)"""
        self._prefill_tokens[p_idx] = max(
            0, self._prefill_tokens.get(p_idx, 0) - isl_tokens)

    def free(self, p_idx: int, d_idx: int, block_count: int) -> None:
        self._decode_blocks[d_idx] = max(
            0, self._decode_blocks.get(d_idx, 0) - block_count)
        reqs = self._prefill_requests.get(p_idx, 0)
        self._prefill_requests[p_idx] = max(0, reqs - 1)

    def snapshot(self, worker_id: int) -> WorkerLoad:
        """One lookup per score (official: request-specific projection)."""
        return WorkerLoad(
            active_prefill_tokens=self._prefill_tokens.get(worker_id, 0),
            active_decode_blocks=self._decode_blocks.get(worker_id, 0),
            active_requests=self._prefill_requests.get(worker_id, 0),
        )

    # Convenience accessors used by eligible() filtering.
    def active_prefill_tokens(self, worker_id: int) -> int:
        return self._prefill_tokens.get(worker_id, 0)

    def decode_usage(self, worker_id: int) -> int:
        return self._decode_blocks.get(worker_id, 0)
