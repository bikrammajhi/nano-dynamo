from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class KvCacheEntry:
    key_cache: List[torch.Tensor]
    value_cache: List[torch.Tensor]
    block_offset: int
    num_blocks: int
    source_worker_id: int


class KvCacheStore:
    def __init__(self) -> None:
        self._store: Dict[str, KvCacheEntry] = {}
        self._lock = threading.Lock()

    def store(
        self,
        request_id: str,
        key_cache: List[torch.Tensor],
        value_cache: List[torch.Tensor],
        block_offset: int,
        num_blocks: int,
        source_worker_id: int,
    ) -> None:
        with self._lock:
            self._store[request_id] = KvCacheEntry(
                key_cache=key_cache,
                value_cache=value_cache,
                block_offset=block_offset,
                num_blocks=num_blocks,
                source_worker_id=source_worker_id,
            )
            logger.debug(
                "KV store: request=%s, source=%d, blocks=%d, offset=%d",
                request_id, source_worker_id, num_blocks, block_offset,
            )

    def load(self, request_id: str) -> Optional[KvCacheEntry]:
        with self._lock:
            return self._store.pop(request_id, None)

    def peek(self, request_id: str) -> bool:
        with self._lock:
            return request_id in self._store

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._store)


class KvCacheTransfer:
    def __init__(self, store: Optional[KvCacheStore] = None, block_size: int = 16) -> None:
        self.store = store or KvCacheStore()
        self.block_size = block_size

    def extract_kv_cache(
        self,
        llm_engine: object,
        block_offset: int,
        num_blocks: int,
        request_id: str,
        source_worker_id: int,
    ) -> bool:
        try:
            kv_cache = self._get_kv_cache(llm_engine)
            if kv_cache is None:
                logger.warning("Could not locate KV cache in vLLM engine")
                return False

            key_cache: List[torch.Tensor] = []
            value_cache: List[torch.Tensor] = []

            for layer_kv in kv_cache:
                if not isinstance(layer_kv, (tuple, list)) or len(layer_kv) != 2:
                    logger.warning("Unexpected KV cache format: %s", type(layer_kv))
                    return False

                key_tensor, value_tensor = layer_kv
                end_block = min(block_offset + num_blocks, key_tensor.shape[0])
                if block_offset >= end_block:
                    logger.warning("Block offset %d >= total blocks %d", block_offset, key_tensor.shape[0])
                    return False

                key_cache.append(key_tensor[block_offset:end_block].cpu().clone())
                value_cache.append(value_tensor[block_offset:end_block].cpu().clone())

            self.store.store(
                request_id=request_id,
                key_cache=key_cache,
                value_cache=value_cache,
                block_offset=block_offset,
                num_blocks=end_block - block_offset,
                source_worker_id=source_worker_id,
            )
            logger.info(
                "Extracted KV cache: request=%s, layers=%d, blocks=%d, offset=%d, source=%d",
                request_id, len(key_cache), num_blocks, block_offset, source_worker_id,
            )
            return True
        except Exception as e:
            logger.error("Failed to extract KV cache: %s", e)
            return False

    def inject_kv_cache(
        self,
        llm_engine: object,
        request_id: str,
        target_worker_id: int,
    ) -> bool:
        try:
            entry = self.store.load(request_id)
            if entry is None:
                logger.warning("No KV cache found for request %s", request_id)
                return False

            kv_cache = self._get_kv_cache(llm_engine)
            if kv_cache is None:
                logger.warning("Could not locate KV cache in vLLM engine")
                return False

            target_device = self._get_device(llm_engine)
            for i, layer_kv in enumerate(kv_cache):
                if i >= len(entry.key_cache):
                    break
                key_tensor, value_tensor = layer_kv
                key_slice = entry.key_cache[i].to(device=target_device)
                value_slice = entry.value_cache[i].to(device=target_device)
                end_block = min(entry.block_offset + entry.num_blocks, key_tensor.shape[0])
                num_to_copy = end_block - entry.block_offset
                key_tensor[entry.block_offset:end_block] = key_slice[:num_to_copy]
                value_tensor[entry.block_offset:end_block] = value_slice[:num_to_copy]

            logger.info(
                "Injected KV cache: request=%s, layers=%d, blocks=%d, offset=%d, target=%d",
                request_id, len(entry.key_cache), entry.num_blocks, entry.block_offset, target_worker_id,
            )
            return True
        except Exception as e:
            logger.error("Failed to inject KV cache: %s", e)
            return False

    def transfer_kv_cache(
        self,
        source_engine: object,
        target_engine: object,
        request_id: str,
        block_offset: int,
        num_blocks: int,
        source_worker_id: int,
        target_worker_id: int,
    ) -> bool:
        if not self.extract_kv_cache(source_engine, block_offset, num_blocks, request_id, source_worker_id):
            return False
        if not self.inject_kv_cache(target_engine, request_id, target_worker_id):
            return False
        logger.info(
            "KV transfer complete: request=%s, %d blocks, worker %d -> %d",
            request_id, num_blocks, source_worker_id, target_worker_id,
        )
        return True

    def can_transfer(self, engine: object) -> bool:
        return self._get_kv_cache(engine) is not None

    def _get_kv_cache(self, llm_engine: object) -> Optional[List]:
        kv_cache = getattr(llm_engine, "kv_cache", None)
        if kv_cache is not None:
            return kv_cache

        inner = getattr(llm_engine, "llm_engine", None)
        if inner is not None:
            kv_cache = getattr(inner, "kv_cache", None)
            if kv_cache is not None:
                return kv_cache
            model_runner = getattr(inner, "model_runner", None)
            if model_runner is not None:
                kv_cache = getattr(model_runner, "kv_cache", None)
                if kv_cache is not None:
                    return kv_cache

        engine_core = getattr(llm_engine, "_engine_core", None)
        if engine_core is not None:
            kv_cache = getattr(engine_core, "kv_cache", None)
            if kv_cache is not None:
                return kv_cache

        return None

    def _get_device(self, llm_engine: object) -> torch.device:
        try:
            inner = getattr(llm_engine, "llm_engine", llm_engine)
            model_runner = getattr(inner, "model_runner", None)
            if model_runner is not None:
                model = getattr(model_runner, "model", None)
                if model is not None:
                    for param in model.parameters():
                        return param.device
        except Exception:
            pass
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")


_global_store: Optional[KvCacheStore] = None
_global_store_lock = threading.Lock()


def get_kv_cache_store() -> KvCacheStore:
    global _global_store
    with _global_store_lock:
        if _global_store is None:
            _global_store = KvCacheStore()
        return _global_store
