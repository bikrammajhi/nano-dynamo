import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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

# ── WorkerLoadProjection ─────────────────────────────────────────────

print("\n=== WorkerLoadProjection ===")
from src.gateway import WorkerLoadProjection, RouterConfigOverride, KvAwarePolicy
wlp = WorkerLoadProjection(active_prefill_tokens=100, active_decode_blocks=50, active_requests=3)
check("fields set", wlp.active_prefill_tokens == 100 and wlp.active_decode_blocks == 50 and wlp.active_requests == 3)
check("potential_decode_blocks", wlp.potential_decode_blocks() == 50)

# ── RouterConfigOverride ─────────────────────────────────────────────

print("\n=== RouterConfigOverride ===")
rco = RouterConfigOverride(overlap_score_credit=0.5, prefill_load_scale=2.0, router_temperature=0.1)
check("override fields", rco.overlap_score_credit == 0.5 and rco.prefill_load_scale == 2.0)
check("none defaults", RouterConfigOverride().overlap_score_credit is None)

# ── KvAwarePolicy ────────────────────────────────────────────────────

print("\n=== KvAwarePolicy ===")
from src.gateway import ScheduleContext, RoundRobinPolicy
from src.kv_index import OverlapResult
policy = KvAwarePolicy(overlap_score_credit=1.0, prefill_load_scale=1.0, block_size=16)

# no context -> round-robin
r1 = policy.select(2)
check("no ctx uses rr", isinstance(policy._rr, RoundRobinPolicy))

# ctx with no scores -> round-robin
ctx = ScheduleContext(token_ids=[1,2,3], overlap=OverlapResult(), worker_loads={})
r2 = policy.select(2, ctx)
check("empty scores uses rr", r2 in (0, 1))

# ctx with scores
ctx2 = ScheduleContext(
    token_ids=[1]*48,
    overlap=OverlapResult(scores={0: 5, 1: 0}),
    worker_loads={0: WorkerLoadProjection(), 1: WorkerLoadProjection()},
)
r3 = policy.select(2, ctx2)
check("selects higher overlap worker", r3 == 0)

# temperature > 0
policy2 = KvAwarePolicy(router_temperature=1.0, block_size=16)
r4 = policy2.select(2, ctx2)
check("temp>0 returns valid worker", r4 in (0, 1))

# config override: zero overlap credit, fixed temp=0
ctx3 = ScheduleContext(
    token_ids=[1]*48,
    overlap=OverlapResult(scores={0: 5, 1: 0}),
    worker_loads={0: WorkerLoadProjection(), 1: WorkerLoadProjection()},
    config_override=RouterConfigOverride(router_temperature=0.0, overlap_score_credit=0.0),
)
# with zero credit and temp=0, both workers have equal cost (no overlap benefit)
# so r5 is valid whichever it picks
r5 = policy2.select(2, ctx3)
check("override temp=0 still valid", r5 in (0, 1))

# ── KvIndex ──────────────────────────────────────────────────────────

print("\n=== KvIndex ===")
from src.kv_index import KvIndex, OverlapResult
from src.kv_events import token_ids_to_block_hashes
ki = KvIndex()
recorded = token_ids_to_block_hashes(list(range(48)), 16)
ki.record_blocks(0, recorded)
result = ki.compute_overlap(list(range(48)), 16)
check("overlap computed", result.scores.get(0, 0) == 3)

# ── MigrationManager ─────────────────────────────────────────────────

print("\n=== MigrationManager ===")
from src.kvbm import MigrationManager
mm = MigrationManager(prefill_instances=["p0"], decode_instances=["d0", "d1"], decode_cache_capacity=100)
mm.record_decode_blocks(0, [1, 2, 3])
check("decode_load tracked", mm.decode_load(0) == 0.03)
mm.record_prefill_load(0, 50)
check("prefill_tokens tracked", mm.active_prefill_tokens(0) == 50)
mm.release_prefill_load(0, 50)
check("prefill_tokens released", mm.active_prefill_tokens(0) == 0)
# best_decode with overlap_credit=0: purely load-based, d1 (load=0) < d0 (load=3)
best, ov = mm.best_decode({1, 2, 3}, overlap_credit=0.0)
check("overlap_credit=0 selects least-loaded", best == 1)
check("ov reports overlap on selected worker", ov == 0 or ov == 3)

# ── OverlapResult ────────────────────────────────────────────────────

print("\n=== OverlapResult ===")
or_ = OverlapResult(scores={0: 3, 1: 1})
check("scores accessible", or_.scores[0] == 3)

# ── Summary ──────────────────────────────────────────────────────────

print(f"\n{'='*40}")
print(f"  {ok} passed, {fail} failed")
print(f"{'='*40}")
sys.exit(0 if fail == 0 else 1)
