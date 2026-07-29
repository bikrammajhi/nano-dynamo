import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.types import LocalBlockHash, ExternalSequenceBlockHash, BlockId, WorkerId, WorkerLoad
from src.block_pool import BlockPool, KvBlockState
from src.radix_tree import RadixTree, KvIndexer, compute_block_hash_for_seq, KvCacheEventData, KvCacheStoredBlockData, KvCacheEvent, RouterEvent
from src.scheduler import SchedulerConfig, KvScheduler, RequestQueue, QueuedRequest, OverlapSignals
from src.prefill import DisaggregatedRouter

import time

ok = 0
fail = 0

def check(name, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {name}")
    else:
        fail += 1
        print(f"  FAIL  {name}  {detail}")

# ── BlockPool ────────────────────────────────────────────────────────

print("\n=== BlockPool ===")
pool = BlockPool(total_blocks=10, block_size=64)
check("init free=10", len(pool._free) == 10)
check("init committed=0", len(pool._committed_index) == 0)

# allocate
bids = pool.allocate(3)
check("allocate returns 3 ids", len(bids) == 3)
check("free now 7", len(pool._free) == 7)

# commit
h = [LocalBlockHash(i) for i in range(3)]
for b, hh in zip(bids, h):
    pool.commit(b, hh)
check("committed=3 after commit", len(pool._committed_index) == 3)
check("free still 7 (committed != free)", len(pool._free) == 7)

# duplicate hash commit (dedup via new block)
new_bid, = pool.allocate(1)
pool.commit(new_bid, h[0])
check("committed still 3 after dedup (same hash)", len(pool._committed_index) == 3)

# match
m = pool.match_blocks([LocalBlockHash(0), LocalBlockHash(1), LocalBlockHash(99)])
check("match cached=2", m.cached_count == 2)
check("match inflight=0", m.inflight_count == 0)
check("match net_new=1", m.net_new_blocks == 1)

# eviction via allocate when full
pool2 = BlockPool(total_blocks=3, block_size=64)
b2 = pool2.allocate(3)
for b, hh in zip(b2, [LocalBlockHash(i) for i in range(3)]):
    pool2.commit(b, hh)
pool2.allocate(1)  # should evict oldest
check("eviction: free=0 after allocate", len(pool2._free) == 0)

# release (only INFLIGHT blocks)
new_bid2, = pool.allocate(1)
pool.release([new_bid2])
check("released 1 inflight block", len(pool._free) == 7)

# allocate 0
z = pool.allocate(0)
check("allocate 0 returns []", z == [])

# negative allocate
try:
    pool.allocate(-1)
    check("negative allocate raises", False)
except ValueError:
    check("negative allocate raises ValueError", True)

# ── RadixTree ────────────────────────────────────────────────────────

print("\n=== RadixTree ===")
tree = RadixTree()
scores = tree.find_matches([LocalBlockHash(0), LocalBlockHash(1)])
check("empty tree returns empty scores", scores.scores == {})

# store event
blocks = [KvCacheStoredBlockData(tokens_hash=LocalBlockHash(i), block_hash=ExternalSequenceBlockHash(i)) for i in range(3)]
event = RouterEvent(worker_id=0, event=KvCacheEvent(event_id=1, data=KvCacheEventData.stored(parent_hash=None, blocks=blocks)))
tree.apply_event(event)
check("worker 0 has 3 entries", len(tree.lookup.get(0, {})) == 3)

# find matches
scores = tree.find_matches([LocalBlockHash(0), LocalBlockHash(1)])
check("matched worker 0 twice", scores.scores.get(0) == 2)

# partial match
scores = tree.find_matches([LocalBlockHash(0), LocalBlockHash(99)])
check("partial match = 1", scores.scores.get(0) == 1)

# remove worker
tree.remove_worker(0)
check("worker 0 removed", 0 not in tree.lookup)
scores = tree.find_matches([LocalBlockHash(0)])
check("no match after removal", scores.scores == {})

# ── KvIndexer ────────────────────────────────────────────────────────

print("\n=== KvIndexer ===")
indexer = KvIndexer(block_size=16)
tok = list(range(48))  # 3 full blocks
computed = compute_block_hash_for_seq(tok, 16)
blocks_i = [KvCacheStoredBlockData(tokens_hash=h, block_hash=ExternalSequenceBlockHash(i)) for i, h in enumerate(computed)]
event_i = RouterEvent(worker_id=0, event=KvCacheEvent(event_id=2, data=KvCacheEventData.stored(parent_hash=None, blocks=blocks_i)))
indexer.apply_event(event_i)
scores = indexer.find_matches_for_request(tok)
check("indexer found matches", scores.scores.get(0, 0) == 3)

# ── Scheduler ────────────────────────────────────────────────────────

print("\n=== Scheduler ===")
sched = KvScheduler(config=SchedulerConfig(overlap_weight=2.0, max_requests_per_worker=16), indexer=None, block_size=64)

# single worker
loads = {WorkerId(0): WorkerLoad(worker_id=WorkerId(0), active_requests=0, kv_cache_used_blocks=0, kv_cache_total_blocks=100)}
r = sched.select_worker([1,2,3], loads)
check("single worker selected 0", r == WorkerId(0))

# temperature softmax
sched2 = KvScheduler(config=SchedulerConfig(overlap_weight=2.0, max_requests_per_worker=16), block_size=64)
r2 = sched2.select_worker([1,2,3], loads)
check("temp>0 still selects valid worker", r2 is not None)

# full worker skipped
loads_full = {WorkerId(0): WorkerLoad(worker_id=WorkerId(0), active_requests=16, kv_cache_used_blocks=0, kv_cache_total_blocks=100)}
r3 = sched2.select_worker([1,2,3], loads_full)
check("full worker returns None", r3 is None)

# empty token ids
r4 = sched.select_worker([], loads)
check("empty tokens selects non-full worker", r4 == WorkerId(0))

# overlap scoring
sched3 = KvScheduler(config=SchedulerConfig(overlap_weight=2.0, max_requests_per_worker=16), indexer=indexer, block_size=16)
loads2 = {WorkerId(0): WorkerLoad(worker_id=WorkerId(0), active_requests=0, kv_cache_used_blocks=0, kv_cache_total_blocks=100),
          WorkerId(1): WorkerLoad(worker_id=WorkerId(1), active_requests=0, kv_cache_used_blocks=0, kv_cache_total_blocks=100)}
r5 = sched3.select_worker(tok, loads2)
# worker 0 has overlap=3, so should be selected over worker 1
check("overlap winner is worker 0", r5 == WorkerId(0))

# ── OverlapSignals ───────────────────────────────────────────────────

print("\n=== OverlapSignals ===")
osig = OverlapSignals()
check("empty signals", osig.overlap_blocks == {} and osig.effective_cached_tokens == {})
osig.overlap_blocks[0] = 5
osig.effective_cached_tokens[0] = 5 * 64
check("assigned overlap 5", osig.overlap_blocks[0] == 5)

# ── RequestQueue ─────────────────────────────────────────────────────

print("\n=== RequestQueue ===")
q = RequestQueue(max_size=3)
check("empty queue false", not q)
check("len=0", len(q) == 0)
rq = QueuedRequest(token_ids=[1], worker_loads={WorkerId(0): WorkerLoad(worker_id=WorkerId(0))})
check("enqueue ok", q.enqueue(rq))
check("len=1", len(q) == 1)
check("non-empty true", bool(q))
# fill to max
q.enqueue(QueuedRequest(token_ids=[2], worker_loads={}))
q.enqueue(QueuedRequest(token_ids=[3], worker_loads={}))
check("full queue", len(q) == 3)
check("enqueue on full returns False", not q.enqueue(QueuedRequest(token_ids=[4], worker_loads={})))

# ── DisaggregatedRouter ──────────────────────────────────────────────

print("\n=== DisaggregatedRouter ===")
worker_ids = [WorkerId(0), WorkerId(1)]
pool0 = BlockPool(total_blocks=100, block_size=64)
pool1 = BlockPool(total_blocks=100, block_size=64)
disagg = DisaggregatedRouter(worker_ids=worker_ids, block_pools={WorkerId(0): pool0, WorkerId(1): pool1}, block_size=64, max_local_prefill_length=512)

# short prefix -> local
dec = disagg.should_prefill_remote([1,2,3], WorkerId(0), {WorkerId(0): 0, WorkerId(1): 0})
check("short prefix: local", not dec.should_prefill_remote)

# long prefix -> remote
dec2 = disagg.should_prefill_remote([1]*1000, WorkerId(0), {WorkerId(0): 0, WorkerId(1): 0})
check("long prefix: remote", dec2.should_prefill_remote)
check("remote worker is 1", dec2.remote_worker_id == WorkerId(1))

# deep queue -> remote
disagg.update_queue_depth(WorkerId(0), 5)
dec3 = disagg.should_prefill_remote([1,2,3], WorkerId(0), {WorkerId(0): 5, WorkerId(1): 0})
check("deep queue: remote", dec3.should_prefill_remote)
check("remote worker is 1", dec3.remote_worker_id == WorkerId(1))

# remote worker picks least-loaded candidate
disagg.update_queue_depth(WorkerId(0), 0)
disagg.update_queue_depth(WorkerId(1), 5)
dec4 = disagg.should_prefill_remote([1]*1000, WorkerId(0), {WorkerId(0): 0, WorkerId(1): 5})
check("remote offloads to least-loaded", dec4.should_prefill_remote)
check("falls back to only candidate", dec4.remote_worker_id == WorkerId(1))

# ── Summary ──────────────────────────────────────────────────────────

print(f"\n{'='*40}")
print(f"  {ok} passed, {fail} failed")
print(f"{'='*40}")
sys.exit(0 if fail == 0 else 1)
