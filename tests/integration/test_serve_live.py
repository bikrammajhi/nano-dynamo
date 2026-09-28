"""Live serving test: real Frontend + real PrefillRouter over real HTTP/SSE.

Fake vLLM workers (no GPU, no weights) stand in for engines: prefill answers
the capped producer call, decode emits canned SSE. Everything between client
and workers is production code: preprocess, S3/S6 scoring, dispatch,
streaming, completion accounting, affinity.

Run: PYTHONPATH=src uv run --with fastapi --with httpx --with xxhash \
       --with uvicorn --with pytest --no-project \
       python -m pytest tests/integration/test_serve_live.py -q
"""

import threading
import time

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from frontend.frontend import Frontend, create_app
from infra.discovery import Discovery
from router.sessions import SessionAffinity
from workers.decode_worker import DecodeWorker
from workers.prefill_worker import PrefillWorker

P_PORT, D_PORT, FE_PORT = 19100, 19200, 18787

# What the fakes observed (asserted later).
seen_prefill_bodies = []
seen_decode_bodies = []
prefill_hits = {"n": 0}
decode_hits = {"n": 0}


def make_prefill():
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(raw: Request):
        body = await raw.json()
        seen_prefill_bodies.append(body)
        prefill_hits["n"] += 1
        return JSONResponse({"choices": [{"message": {"content": "p"}}]})

    @app.get("/v1/models")
    async def models():
        return JSONResponse({"data": [{"id": "fake-model"}]})

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


def make_decode():
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(raw: Request):
        body = await raw.json()
        seen_decode_bodies.append(body)
        decode_hits["n"] += 1

        async def gen():
            yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
            yield b"data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


def serve(app, port):
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                           log_level="error"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    return server


def wait_healthy(port, timeout=20):
    start = time.time()
    while time.time() - start < timeout:
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.2)
    raise RuntimeError(f"server on {port} never became healthy")


class StubTok:
    def apply_chat_template(self, msgs, **kw):
        return "shared system prompt words here " * 20

    def encode(self, text):
        return list(range(256))


def build_frontend():
    Frontend._tokenizer = lambda self: StubTok()
    disc = Discovery(
        prefill=[PrefillWorker(url=f"127.0.0.1:{P_PORT}", engine_id="prefill-0",
                               side_port=5600, gpu=0)],
        decode=[DecodeWorker(url=f"127.0.0.1:{D_PORT}", engine_id="decode-0", gpu=1)],
    )
    app = create_app(model="fake", router_mode="kv", discovery=disc)
    fe = app.state.frontend
    fe.router.sessions = SessionAffinity(fe.router.indexer, fe.router.slots)
    return app, fe


def test_live_serve():
    serve(make_prefill(), P_PORT)
    serve(make_decode(), D_PORT)
    app, fe = build_frontend()
    serve(app, FE_PORT)
    wait_healthy(FE_PORT)
    base = f"http://127.0.0.1:{FE_PORT}"

    # 1. basic chat, streamed
    r = httpx.post(f"{base}/v1/chat/completions",
                   json={"model": "fake",
                         "messages": [{"role": "user", "content": "hi"}]},
                   timeout=30)
    assert r.status_code == 200, r.text[:200]
    assert "hi" in r.text
    # producer contract enforced on the wire
    assert seen_prefill_bodies[-1]["max_tokens"] == 1
    assert seen_prefill_bodies[-1]["max_completion_tokens"] == 1
    # decode got transfer metadata
    xfer = seen_decode_bodies[-1]["kv_transfer_params"]
    assert xfer["remote_engine_id"] == "prefill-0"
    assert xfer["remote_port"] == 5600
    time.sleep(0.5)  # background _finish_push + completion

    # 2. multi-turn affinity: same conversation reuses the binding
    conv = {"model": "fake", "messages": [{"role": "user", "content": "t"}],
            "conversation_id": "live-conv-1"}
    assert httpx.post(f"{base}/v1/chat/completions", json=conv, timeout=30).status_code == 200
    time.sleep(0.5)
    bound = fe.router.sessions.lookup("live-conv-1")
    assert bound is not None
    n_before = (prefill_hits["n"], decode_hits["n"])
    assert httpx.post(f"{base}/v1/chat/completions", json=conv, timeout=30).status_code == 200
    time.sleep(0.5)
    assert (prefill_hits["n"], decode_hits["n"]) == (n_before[0] + 1, n_before[1] + 1)
    assert fe.router.sessions.lookup("live-conv-1") is not None

    # 3. KV locality: shared-prefix traffic recorded worker-0 blocks
    assert fe.router.indexer.overlap(0, "prefill", {1, 2}) >= 0
    assert len(fe.router.indexer.worker_blocks(0, "prefill")) > 0

    # 4. observability + validation paths
    assert httpx.get(f"{base}/health", timeout=10).json() == {"status": "ok"}
    assert httpx.get(f"{base}/v1/models", timeout=10).status_code == 200
    ks = httpx.get(f"{base}/kvbm/status", timeout=10)
    assert ks.status_code == 200 and "decode_usage" in ks.json()
    bad = httpx.post(f"{base}/v1/chat/completions", json={"model": "fake"}, timeout=10)
    assert bad.status_code == 400


def test_live_decode_failure_inhibits():
    serve(make_prefill(), P_PORT + 10)
    app, fe = build_frontend()
    # point decode at a dead port: first request 502s, worker gets inhibited,
    # second request 503s (compatible but all filtered) instead of retrying blind
    fe.router.decode_urls = ["127.0.0.1:19999"]
    serve(app, FE_PORT + 10)
    wait_healthy(FE_PORT + 10)
    base = f"http://127.0.0.1:{FE_PORT + 10}"
    body = {"model": "fake", "messages": [{"role": "user", "content": "hi"}]}
    r1 = httpx.post(f"{base}/v1/chat/completions", json=body, timeout=30)
    assert r1.status_code == 502, r1.text[:200]
    assert fe.is_inhibited("http://127.0.0.1:19999")
    r2 = httpx.post(f"{base}/v1/chat/completions", json=body, timeout=30)
    assert r2.status_code == 503, r2.text[:200]
