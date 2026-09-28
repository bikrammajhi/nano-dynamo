"""Yellow box (dotted): KV Events plane + block hashing. See docs/architecture.mmd.

Local stand-in for KVPublisher: instead of stored/removed events over NATS/ZMQ,
hashes feed KvIndexer by direct call. Same information, no transport.
Official counterparts: lib/kv-hashing/, identity.rs, tracking_hash.rs.

Block-management flow steps covered here (cf. Router Design doc):
  1. block partitioning: token stream -> fixed-size blocks
  2. block hashing: xxh3_64 per full block (partial tail dropped - vLLM caches
     full blocks only, so a partial hash would never match anything)
"""

from __future__ import annotations

from typing import List

import xxhash


def token_ids_to_block_hashes(token_ids: List[int], block_size: int) -> List[int]:
    """Partition + hash. Deterministic across processes (fixed seed, no
    hash() randomization) so two workers hash the same prefix identically."""
    hashes: List[int] = []
    for i in range(0, len(token_ids), block_size):
        chunk = token_ids[i:i + block_size]
        if len(chunk) < block_size:
            break
        raw = b"".join(t.to_bytes(4, "little") for t in chunk)
        hashes.append(xxhash.xxh3_64(raw, seed=1337).intdigest())
    return hashes
