"""Utility functions for KV block hashing (used by KV-aware routing)."""
from __future__ import annotations

from typing import List

import xxhash


def token_ids_to_block_hashes(token_ids: List[int], block_size: int) -> List[int]:
    """Compute block hashes from a list of token IDs.

    Only full blocks are hashed (partial last block is dropped).
    """
    hashes: List[int] = []
    for i in range(0, len(token_ids), block_size):
        chunk = token_ids[i : i + block_size]
        if len(chunk) < block_size:
            break
        raw = b"".join(t.to_bytes(4, "little") for t in chunk)
        hashes.append(xxhash.xxh3_64(raw, seed=1337).intdigest())
    return hashes
