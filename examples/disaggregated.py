"""Minimal local wiring: 1P + 1D disaggregated stack (no GPUs needed to build).

Needs real vLLM workers on :8100 (prefill) + :8200 (decode) to SERVE;
building the stack only proves the wiring. Mirrors examples/ purpose in the
official repo: smallest runnable topology, not a benchmark.
"""

import sys

sys.path.insert(0, "src")

from frontend.frontend import create_app
from infra.discovery import Discovery
from router.sessions import SessionAffinity
from workers.decode_worker import DecodeWorker
from workers.prefill_worker import PrefillWorker

disc = Discovery(
    prefill=[PrefillWorker(url="127.0.0.1:8100", engine_id="prefill-0",
                           side_port=5600, gpu=0)],
    decode=[DecodeWorker(url="127.0.0.1:8200", engine_id="decode-0", gpu=1)],
)
app = create_app(model="Qwen/Qwen3-14B-FP8", router_mode="kv", discovery=disc)
fe = app.state.frontend
fe.router.sessions = SessionAffinity(fe.router.indexer, fe.router.slots)

print("wired:", disc.service_discovery())
print("models source:", fe.models_source_url)
print("serve with: uvicorn frontend.frontend:<app> --port 8787  (needs vLLM up)")
