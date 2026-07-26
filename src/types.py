# SPDX-FileCopyrightText: Copyright (c) 2025 nano-dynamo contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set


_XXH3_SEED: int = 1337


def _compute_hash(data: bytes) -> int:
    import xxhash
    return xxhash.xxh3_64(data, seed=_XXH3_SEED).intdigest()


@dataclass(frozen=True)
class LocalBlockHash:
    value: int

    def __hash__(self) -> int:
        return self.value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, LocalBlockHash) and self.value == other.value

    @staticmethod
    def from_token_ids(token_ids: bytes) -> LocalBlockHash:
        return LocalBlockHash(_compute_hash(token_ids))


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


@dataclass(frozen=True)
class RequestId:
    value: str

    def __hash__(self) -> int:
        return hash(self.value)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RequestId) and self.value == other.value


class FinishReason(str, Enum):
    EoS = "eos"
    Length = "length"
    Stop = "stop"
    Error = "error"
    Cancelled = "cancelled"


@dataclass
class StopConditions:
    max_tokens: Optional[int] = None
    stop: Optional[List[str]] = None
    stop_token_ids: Optional[List[int]] = None
    min_tokens: Optional[int] = None
    ignore_eos: bool = False

    def apply_ignore_eos(self) -> None:
        if self.ignore_eos:
            self.min_tokens = self.max_tokens
            self.stop = None
            self.stop_token_ids = None


@dataclass
class SamplingOptions:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    seed: Optional[int] = None

    def force_greedy(self) -> None:
        self.temperature = 0.0
        self.top_p = 1.0
        self.top_k = -1


@dataclass
class Request:
    request_id: RequestId
    worker_id: WorkerId
    prompt: str
    stop: StopConditions
    sampling: SamplingOptions
    token_ids: Optional[List[int]] = None


@dataclass
class Delta:
    token_id: int
    text: str
    is_complete: bool = False
    finish_reason: Optional[FinishReason] = None


@dataclass
class Response:
    request_id: RequestId
    text: str
    finish_reason: FinishReason
    stats: "Stats"


@dataclass
class Stats:
    request_active_count: int = 0
    request_context_count: int = 0
    request_generation_count: int = 0
    request_scheduled_count: int = 0
    request_max_count: int = 0
    kv_free_cache_blocks: int = 0
    kv_max_cache_blocks: int = 0
    kv_used_cache_blocks: int = 0
    kv_tokens_per_cache_block: int = 0
    runtime_cpu_memory_usage: int = 0
    runtime_gpu_memory_usage: int = 0
    runtime_pinned_memory_usage: int = 0
    iteration_counter: int = 0
    total_context_tokens: int = 0
    output_tokens: int = 0
    timestamp: str = ""


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

    @property
    def is_busy(self) -> bool:
        return self.kv_cache_used_blocks >= self.kv_cache_total_blocks
