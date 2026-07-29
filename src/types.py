# SPDX-FileCopyrightText: Copyright (c) 2025 kv-prefix-router contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LocalBlockHash:
    value: int

    def __hash__(self) -> int:
        return self.value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, LocalBlockHash) and self.value == other.value


@dataclass(frozen=True)
class ExternalSequenceBlockHash:
    value: int

    def __hash__(self) -> int:
        return self.value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ExternalSequenceBlockHash) and self.value == other.value


@dataclass(frozen=True)
class BlockId:
    value: int

    def __hash__(self) -> int:
        return self.value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BlockId) and self.value == other.value


@dataclass(frozen=True)
class WorkerId:
    value: int

    def __hash__(self) -> int:
        return self.value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, WorkerId) and self.value == other.value


@dataclass
class WorkerLoad:
    worker_id: WorkerId
    active_requests: int = 0
    queued_requests: int = 0
    kv_cache_used_blocks: int = 0
    kv_cache_total_blocks: int = 0
    prefix_cache_queries: int = 0
    prefix_cache_hits: int = 0
    prefix_cache_hit_rate: float = 0.0
    active_prefill_tokens: int = 0
    active_decode_blocks: int = 0

    def potential_decode_blocks(self) -> int:
        return self.active_decode_blocks
