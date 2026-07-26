from __future__ import annotations
import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, AsyncGenerator, List, Optional, Set
from .types import (
    BlockId, FinishReason, LocalBlockHash, Request, RequestId,
    Response, Stats, StopConditions, WorkerId, WorkerLoad,
)
from .block_pool import BlockPool, PrefillMatched
from .radix_tree import (
    KvCacheEvent, KvCacheEventData, KvCacheStoredBlockData,
    KvIndexer, RouterEvent,
)
if TYPE_CHECKING:
    from .prefill import DisaggregatedRouter
    from .kv_transfer import KvCacheTransfer
logger = logging.getLogger(__name__)

@dataclass
class Decoder:
    stop_conditions: StopConditions
    generated_tokens: int = 0
    hidden_stop_ids: Set[int] = field(default_factory=set)
    hidden_stop_sequences: List[str] = field(default_factory=list)
    jail: str = ""
    jail_max_bytes: int = 0
    @classmethod
    def from_stop_conditions(cls, stop: StopConditions) -> Decoder:
        hids: Set[int] = set(stop.stop_token_ids or [])
        hseqs: List[str] = [s for s in (stop.stop or []) if s]
        return cls(stop_conditions=stop, hidden_stop_ids=hids, hidden_stop_sequences=hseqs, jail_max_bytes=max((len(s) for s in hseqs), default=0))
    def update(self, token_id: int, text: str) -> Optional[StepResult]:
        self.generated_tokens += 1
        if self.generated_tokens < (self.stop_conditions.min_tokens or 0):
            return None
        if token_id in self.hidden_stop_ids:
            return StepResult(text=text, stop_trigger=StopTrigger.HIDDEN_STOP_TOKEN)
        if self.hidden_stop_sequences and text:
            self.jail += text
            if len(self.jail) > self.jail_max_bytes:
                self.jail = self.jail[-self.jail_max_bytes:]
            for seq in self.hidden_stop_sequences:
                if seq in self.jail:
                    return StepResult(text=self.jail[:self.jail.index(seq)], stop_trigger=StopTrigger.HIDDEN_STOP_SEQUENCE)
        return None
    def reset(self) -> None:
        self.generated_tokens = 0
        self.jail = ""

class StopTrigger:
    MAX_TOKENS_LIMIT = "max_tokens"
    HIDDEN_STOP_TOKEN = "hidden_stop_token"
    HIDDEN_STOP_SEQUENCE = "hidden_stop_sequence"

@dataclass
class StepResult:
    text: str
    stop_trigger: Optional[str] = None

class Worker:
    def __init__(
        self, worker_id: WorkerId, gpu_id: int, model: str,
        tokenizer_name: Optional[str] = None, scheduler: Optional[KvIndexer] = None,
        block_pool: Optional[BlockPool] = None, max_kv_cache_blocks: int = 256,
        max_active_requests: int = 16, block_size: int = 16,
        disagg_router: Optional[DisaggregatedRouter] = None,
        kv_transfer: Optional[KvCacheTransfer] = None,
    ) -> None:
        self.worker_id = worker_id
        self.gpu_id = gpu_id
        self.model = model
        self.tokenizer_name = tokenizer_name or model
        self.scheduler = scheduler
        self.block_size = block_size
        self.max_kv_cache_blocks = max_kv_cache_blocks
        self.max_active_requests = max_active_requests
        self.disagg_router = disagg_router
        self.kv_transfer = kv_transfer
        self._llm = None
        self._tokenizer = None
        self.block_pool = block_pool or BlockPool(total_blocks=max_kv_cache_blocks, block_size=block_size)
        self._active_requests: int = 0
    def _ensure_initialized(self) -> None:
        if self._llm is not None:
            return
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        from vllm import LLM, SamplingParams
        self._llm = LLM(model=self.model, tokenizer=self.tokenizer_name, gpu_memory_utilization=0.85, max_model_len=8192, block_size=self.block_size, dtype="auto")
        self._sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=512)
        logger.info("worker %d (GPU %d) initialized with model %s", self.worker_id.value, self.gpu_id, self.model)
    @property
    def load(self) -> WorkerLoad:
        kv_used = len(self.block_pool._committed_index)
        return WorkerLoad(worker_id=self.worker_id, active_requests=self._active_requests, kv_cache_used_blocks=kv_used, kv_cache_total_blocks=self.max_kv_cache_blocks)
    def _make_sampling_params(self, request: Request) -> object:
        from vllm import SamplingParams
        return SamplingParams(temperature=request.sampling.temperature, top_p=request.sampling.top_p, max_tokens=request.stop.max_tokens or 512, stop=request.stop.stop or [], stop_token_ids=request.stop.stop_token_ids or [])
    def _commit_and_publish(self, allocated: List[BlockId], token_ids: Optional[List[int]], block_offset: int = 0, do_publish: bool = True) -> bool:
        committed = False
        if token_ids:
            from .radix_tree import compute_block_hash_for_seq
            block_hashes = compute_block_hash_for_seq(token_ids, self.block_size)
            for i, bid in enumerate(allocated):
                if i + block_offset < len(block_hashes):
                    self.block_pool.commit(bid, block_hashes[i + block_offset])
            committed = True
        if do_publish and self.scheduler and token_ids:
            self.publish_kv_events(allocated, token_ids=token_ids, block_offset=block_offset)
        return committed
    async def _decode_tokens(self, request: Request, output_token_ids, generated_text, decoder: Decoder):
        try:
            tok = self._llm.get_tokenizer()
            token_texts = [tok.convert_tokens_to_string([tok.convert_ids_to_tokens([tid])[0]]) for tid in output_token_ids]
            token_texts = [t for t in token_texts if t]
        except Exception:
            token_texts = list(generated_text)
        for tt in token_texts:
            result = decoder.update(0, tt)
            yield Response(request_id=request.request_id, text=tt, finish_reason=FinishReason.STOP if result is not None else None, stats=Stats(output_tokens=len(output_token_ids)))
            if result is not None:
                break
        if not token_texts and generated_text:
            yield Response(request_id=request.request_id, text=generated_text, finish_reason=FinishReason.STOP, stats=Stats(output_tokens=len(output_token_ids), **self.stats().__dict__))
    def run_prefill(self, request: Request, matched: PrefillMatched) -> List[BlockId]:
        self._ensure_initialized()
        net_new = matched.net_new_blocks
        if net_new <= 0:
            return []
        allocated = self.block_pool.allocate(net_new)
        try:
            token_ids = request.token_ids or []
            new_token_ids = token_ids[matched.cached_count * self.block_size:]
            if not new_token_ids:
                return allocated
            from vllm import SamplingParams
            sampling = SamplingParams(temperature=request.sampling.temperature, top_p=request.sampling.top_p, max_tokens=1)
            self._llm.generate([request.prompt], sampling)
            from .radix_tree import compute_block_hash_for_seq
            block_hashes = compute_block_hash_for_seq(token_ids, self.block_size)
            for i, bid in enumerate(allocated):
                hi = matched.cached_count + i
                bh = block_hashes[hi] if hi < len(block_hashes) else LocalBlockHash.from_token_ids(str(bid.value).encode())
                self.block_pool.commit(bid, bh)
            if self.scheduler:
                self.publish_kv_events(allocated, token_ids=token_ids, block_offset=matched.cached_count)
            return allocated
        except Exception:
            self.block_pool.release(allocated)
            raise
    def publish_kv_events(self, allocated: List[BlockId], token_ids: Optional[List[int]] = None, block_offset: int = 0) -> None:
        if not self.scheduler:
            return
        from .types import ExternalSequenceBlockHash
        from .radix_tree import compute_block_hash_for_seq, compute_hash
        block_hashes = compute_block_hash_for_seq(token_ids, self.block_size) if token_ids is not None else [LocalBlockHash.from_token_ids(str(bid.value).encode()) for bid in allocated]
        parent_hash: Optional[ExternalSequenceBlockHash] = None
        blocks: List[KvCacheStoredBlockData] = []
        for i, bid in enumerate(allocated):
            hi = block_offset + i
            bh = block_hashes[hi] if hi < len(block_hashes) else LocalBlockHash.from_token_ids(str(bid.value).encode())
            seq_bytes = bh.value.to_bytes(8, "little") + (parent_hash.value.to_bytes(8, "little") if parent_hash is not None else b"")
            ext_hash = ExternalSequenceBlockHash(compute_hash(seq_bytes))
            blocks.append(KvCacheStoredBlockData(tokens_hash=bh, block_hash=ext_hash))
            parent_hash = ext_hash
        self.scheduler.apply_event(RouterEvent(worker_id=self.worker_id.value, event=KvCacheEvent(event_id=id(self), data=KvCacheEventData.stored(parent_hash=None, blocks=blocks))))
    async def prefill_only(self, request: Request, matched: PrefillMatched) -> List[BlockId]:
        self._ensure_initialized()
        net_new = matched.net_new_blocks
        if net_new <= 0:
            return []
        allocated = self.block_pool.allocate(net_new)
        try:
            token_ids = request.token_ids or []
            new_token_ids = token_ids[matched.cached_count * self.block_size:]
            if not new_token_ids:
                return allocated
            from vllm import SamplingParams
            sampling = SamplingParams(temperature=request.sampling.temperature, top_p=request.sampling.top_p, max_tokens=1)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: self._llm.generate([request.prompt], sampling))
            from .radix_tree import compute_block_hash_for_seq
            block_hashes = compute_block_hash_for_seq(token_ids, self.block_size)
            for i, bid in enumerate(allocated):
                hi = matched.cached_count + i
                bh = block_hashes[hi] if hi < len(block_hashes) else LocalBlockHash.from_token_ids(str(bid.value).encode())
                self.block_pool.commit(bid, bh)
            if self.scheduler:
                self.publish_kv_events(allocated, token_ids=token_ids, block_offset=matched.cached_count)
            return allocated
        except Exception:
            self.block_pool.release(allocated)
            raise
    async def generate(
        self, request: Request, sampling_params: Optional[object] = None,
        disagg_decision: Optional[DisaggregationDecision] = None,
        remote_worker: Optional[Worker] = None,
    ) -> AsyncGenerator[Response, None]:
        self._ensure_initialized()
        can_disagg = (
            disagg_decision is not None and disagg_decision.should_prefill_remote
            and remote_worker is not None and self.kv_transfer is not None
            and self._llm is not None and remote_worker._llm is not None
            and self.kv_transfer.can_transfer(remote_worker._llm)
        )
        gen = self._generate_disagg(request, sampling_params, disagg_decision, remote_worker) if can_disagg else self._generate_local(request, sampling_params)
        async for resp in gen:
            yield resp
    async def _generate_local(self, request: Request, sampling_params: Optional[object] = None) -> AsyncGenerator[Response, None]:
        if sampling_params is None:
            sampling_params = self._make_sampling_params(request)
        token_ids = request.token_ids or []
        num_blocks = max(1, -(-len(token_ids) // self.block_size)) if token_ids else 1
        allocated = self.block_pool.allocate(num_blocks)
        self._active_requests += 1
        committed = False
        try:
            decoder = Decoder.from_stop_conditions(request.stop)
            decoder.reset()
            loop = asyncio.get_event_loop()
            request_output = await loop.run_in_executor(None, lambda: list(self._llm.generate([request.prompt], sampling_params))[0])
            completion = request_output.outputs[0]
            async for resp in self._decode_tokens(request, completion.token_ids or [], completion.text or "", decoder):
                yield resp
            committed = self._commit_and_publish(allocated, token_ids)
        finally:
            self._active_requests = max(0, self._active_requests - 1)
            if not committed:
                self.block_pool.release(allocated)
    async def _generate_disagg(
        self, request: Request, sampling_params: Optional[object],
        disagg_decision: DisaggregationDecision, remote_worker: Worker,
    ) -> AsyncGenerator[Response, None]:
        if sampling_params is None:
            sampling_params = self._make_sampling_params(request)
        token_ids = request.token_ids or []
        self._active_requests += 1
        try:
            if self.kv_transfer and self._llm and remote_worker._llm:
                if not self.kv_transfer.can_transfer(remote_worker._llm):
                    async for resp in self._generate_local(request, sampling_params):
                        yield resp
                    return
            from .radix_tree import compute_block_hash_for_seq
            block_hashes = compute_block_hash_for_seq(token_ids, self.block_size)
            matched = remote_worker.block_pool.match_blocks(block_hashes)
            remote_allocated = await remote_worker.prefill_only(request, matched)
            if remote_allocated and self.kv_transfer and self._llm:
                transfer_success = self.kv_transfer.transfer_kv_cache(
                    source_engine=remote_worker._llm, target_engine=self._llm,
                    request_id=request.request_id.value, block_offset=matched.cached_count,
                    num_blocks=len(remote_allocated), source_worker_id=remote_worker.worker_id.value,
                    target_worker_id=self.worker_id.value,
                )
                if not transfer_success:
                    async for resp in self._generate_local(request, sampling_params):
                        yield resp
                    return
            decoder = Decoder.from_stop_conditions(request.stop)
            decoder.reset()
            loop = asyncio.get_event_loop()
            request_output = await loop.run_in_executor(None, lambda: list(self._llm.generate([request.prompt], sampling_params))[0])
            completion = request_output.outputs[0]
            async for resp in self._decode_tokens(request, completion.token_ids or [], completion.text or "", decoder):
                yield resp
        finally:
            self._active_requests = max(0, self._active_requests - 1)
    def stats(self) -> Stats:
        kv_used = len(self.block_pool._committed_index)
        return Stats(
            request_active_count=self._active_requests, request_max_count=self.max_active_requests,
            kv_free_cache_blocks=self.max_kv_cache_blocks - kv_used, kv_max_cache_blocks=self.max_kv_cache_blocks,
            kv_used_cache_blocks=kv_used, kv_tokens_per_cache_block=self.block_size, timestamp="",
        )
