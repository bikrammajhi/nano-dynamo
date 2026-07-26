from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from .types import WorkerId
from .engine import Worker

if TYPE_CHECKING:
    from .kv_transfer import KvCacheTransfer

logger = logging.getLogger(__name__)


@dataclass
class DisaggregationDecision:
    should_prefill_remote: bool
    remote_worker_id: Optional[WorkerId]
    local_worker_id: WorkerId
    effective_prefill_length: int


class DisaggregatedRouter:
    def __init__(
        self,
        workers: dict[WorkerId, Worker],
        scheduler: Optional["KvScheduler"] = None,
        block_pools: Optional[dict[WorkerId, "BlockPool"]] = None,
        max_local_prefill_length: int = 512,
        prefill_queue_depth_threshold: int = 4,
        block_size: int = 16,
        kv_transfer: Optional["KvCacheTransfer"] = None,
    ) -> None:
        self.workers = workers
        self.scheduler = scheduler
        self.block_pools = block_pools or {}
        self.max_local_prefill_length = max_local_prefill_length
        self.prefill_queue_depth_threshold = prefill_queue_depth_threshold
        self.block_size = block_size
        self.kv_transfer = kv_transfer
        self._prefill_queue_depths: dict[WorkerId, int] = {wid: 0 for wid in workers}

    def should_prefill_remote(
        self,
        request: "Request",
        local_worker_id: WorkerId,
        overlap: Optional[OverlapScores] = None,
    ) -> DisaggregationDecision:
        token_ids = request.token_ids or []
        total_tokens = len(token_ids)

        prefix_hit_tokens = 0
        if overlap and self.block_pools:
            pool = self.block_pools.get(local_worker_id)
            if pool:
                matched = pool.match_blocks(self._blocks_from_tokens(token_ids))
                prefix_hit_tokens = matched.cached_count * self.block_size

        effective_prefill = max(0, total_tokens - prefix_hit_tokens)

        local_queue_depth = self._prefill_queue_depths.get(local_worker_id, 0)
        should_remote = (
            effective_prefill > self.max_local_prefill_length
            or local_queue_depth >= self.prefill_queue_depth_threshold
        )

        if not should_remote:
            return DisaggregationDecision(
                should_prefill_remote=False,
                remote_worker_id=None,
                local_worker_id=local_worker_id,
                effective_prefill_length=effective_prefill,
            )

        remote_worker_id = self._pick_remote_prefill_worker(local_worker_id)
        if remote_worker_id is None:
            return DisaggregationDecision(
                should_prefill_remote=False,
                remote_worker_id=None,
                local_worker_id=local_worker_id,
                effective_prefill_length=effective_prefill,
            )

        return DisaggregationDecision(
            should_prefill_remote=True,
            remote_worker_id=remote_worker_id,
            local_worker_id=local_worker_id,
            effective_prefill_length=effective_prefill,
        )

    def _pick_remote_prefill_worker(self, exclude: WorkerId) -> Optional[WorkerId]:
        candidates = [wid for wid in self.workers if wid != exclude]
        if not candidates:
            return None
        candidates.sort(
            key=lambda wid: (
                self._prefill_queue_depths.get(wid, 0),
                self.workers[wid].load.active_requests,
            )
        )
        return candidates[0]

    def _blocks_from_tokens(self, token_ids: list[int]) -> list["LocalBlockHash"]:
        from .radix_tree import compute_block_hash_for_seq
        return compute_block_hash_for_seq(token_ids, self.block_size)


from .block_pool import BlockPool, PrefillMatched  # noqa: E402
