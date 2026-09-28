"""Diagram: Frontend box (orange) - badges S1, S2, S9 - see docs/architecture.mmd.

The Frontend is the API gateway: OpenAI-compatible ingress (S1), request
preprocessing (S2), and response relay (S9). It decides NOTHING about worker
placement - every routing call goes to PrefillRouter (router/ layer).
Rule: this module never imports the cost function (asserted in test_naming.py).

Feature matrix (cf. official Frontend doc - HAVE vs OUT-OF-SCOPE):
  HAVE:  POST /v1/chat/completions, POST /v1/completions, GET /v1/models,
         streaming SSE, kv/round-robin/random routing (via --router-mode)
  OUT:   /v1/embeddings (no embedding workers), /v1/responses|images (no
         backends for them), nvext headers (no agent traffic), TLS (terminated
         outside), K8s/CRDs (static discovery - see infra/discovery.py),
         Prometheus /metrics (lightweight JSON /kvbm/status instead - no new dep).

Launch: python -m frontend.frontend --http-port 8787 --router-mode kv
(mirror of `python -m dynamo.frontend --http-port 8000`; port differs to avoid
clashing with local vLLM workers, flag names match official CLI.)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import threading
import time
import uuid

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from infra.kv_events import token_ids_to_block_hashes
from router.prefill_router import Route, RoutingError
from workers.prefill_worker import producer_body

log = logging.getLogger("mini-dynamo.frontend")

# One POST primitive for both sinks (merge of old _forward_stream/_drain):
# eager-POST upstream, return an async chunk iterator. Draining discards it,
# streaming relays it. A single transport path means one timeout/close policy.
HTTP_TIMEOUT = httpx.Timeout(600.0, connect=10.0)
# Slow decodes (128 tokens, low concurrency) are normal traffic, not stuck
# requests - keep the total generous, the connect strict.

# Local worker inhibition: after a routed request fails, quarantine that
# worker URL briefly while "discovery catches up". Official counterpart:
# DYN_RUNTIME_INHIBITED_DURATION_SECS (default 5 s). Cheapest real resilience
# this gateway can own (~10 lines): no health-check subsystem, no registry,
# just a timestamp table the router's eligible() consults (Block 5).
INHIBIT_SECS = 5.0

# Tokenizer singleton: process-global, thread-safe lazy load.
# Needs model CONFIGS only (config.json/tokenizer.json) - never weights,
# per the official Frontend note. First preprocess call pays the download.
_tokenizer = None
_tokenizer_lock = threading.Lock()


def _get_tokenizer(model_name: str):
    global _tokenizer
    if _tokenizer is None:
        with _tokenizer_lock:
            if _tokenizer is None:
                from transformers import AutoTokenizer
                _tokenizer = AutoTokenizer.from_pretrained(
                    model_name, trust_remote_code=True
                )
    return _tokenizer

ROUTER_MODES = ("round-robin", "random", "kv")
# Official --router-mode also accepts power-of-two, least-loaded, direct,
# device-aware-weighted. Those fail fast in create_app() with a pointer to
# the official doc - an explicit error beats a silent wrong-strategy fallback.


class UpstreamError(RuntimeError):
    """A worker answered non-200. Carries status + body snippet for the 502."""


class Frontend:
    """Orange box. Owns HTTP + pre/post; delegates ALL placement to PrefillRouter."""

    def __init__(self, model: str, router_mode: str = "kv"):
        if router_mode not in ROUTER_MODES:
            raise NotImplementedError(
                f"--router-mode={router_mode!r} is an official Dynamo mode "
                f"this gateway does not implement "
                f"(see docs/architecture.mmd S3 zoom-in). "
                f"Supported: {ROUTER_MODES}."
            )
        self.model = model
        self.router_mode = router_mode
        # Placement wiring lands with the router layer (Block 5):
        #   self.router   -> PrefillRouter.route(token_ids, session_id) -> Route
        #   self.kv_host  -> side-channel host for S7 metadata
        # Until then, tests inject a stub router; live traffic gets 501.
        self.router = None
        self.prefill_kv_host = "127.0.0.1"
        # Wiring filled with the workers layer (Block 5+): base URL whose
        # /v1/models is proxied (prefill worker 0, cf. old list_models).
        self.models_source_url: str | None = None
        # Inhibition table: worker base URL -> monotonic expiry. See INHIBIT_SECS.
        self._inhibited: dict[str, float] = {}

    @staticmethod
    def _inhibit_key(worker_url: str) -> str:
        # Canonical key: bare host:port. Call sites mix forms (dispatch
        # inhibits "http://h:p", eligible() checks bare "h:p") - normalize
        # once here so writer and reader can never disagree.
        return worker_url.removeprefix("http://").removeprefix("https://")

    def inhibit(self, worker_url: str, secs: float = INHIBIT_SECS) -> None:
        """Quarantine a worker URL until now+secs. Discovery stays authoritative:
        expiry restores the worker automatically; explicit removal is the
        router/discovery layer's job, not this table's."""
        self._inhibited[self._inhibit_key(worker_url)] = time.monotonic() + secs

    def is_inhibited(self, worker_url: str) -> bool:
        """True while quarantined. Lazily drops expired entries on read so the
        table cannot grow with dead URLs over a long-lived process."""
        key = self._inhibit_key(worker_url)
        expiry = self._inhibited.get(key)
        if expiry is None:
            return False
        if time.monotonic() >= expiry:
            del self._inhibited[key]
            return False
        return True

    # ── S1: REQUEST (HTTP API Call) ──────────────────────────
    # S2 (preprocess) below. Dispatch lands in Pass 3.

    # ── S2: PREPROCESS (Tokenize & Validate) ─────────────────
    # Single tokenize per request. The old gateway.py called _tokenize TWICE
    # per request (once in _pick, once in _push on the identical body).
    # preprocess() runs once; its token_ids feed selection AND dispatch.

    def tokenize_and_validate(self, body: dict) -> list[int]:
        """Chat template -> tokenize. 400 on malformed bodies.

        Validation runs BEFORE the tokenizer loads: a bad request must fail
        fast without paying a model-config download. (Old code silently
        yielded empty tokens for missing messages/prompt - that path now 400s.)
        """
        messages = body.get("messages", [])
        prompt = body.get("prompt", "")
        if messages:
            text = self._tokenizer().apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        elif prompt:
            text = prompt
        else:
            raise HTTPException(
                status_code=400,
                detail="body must contain non-empty 'messages' or 'prompt'",
            )
        token_ids = self._tokenizer().encode(text) if text else []
        if not token_ids:
            raise HTTPException(
                status_code=400, detail="prompt tokenizes to zero tokens"
            )
        return token_ids

    def preprocess(self, body: dict) -> list[int]:
        """S2 entry: the ONE tokenization for this request."""
        return self.tokenize_and_validate(body)

    def _tokenizer(self):
        return _get_tokenizer(self.model)

    async def http_api_call(self, body: dict):
        """S1 entry: validate -> preprocess(S2) -> route(S3/S6) -> dispatch."""
        t_req = time.monotonic()
        token_ids = self.preprocess(body)          # S2, the ONE tokenization
        if self.router is None:
            raise HTTPException(status_code=501, detail="router not wired yet")
        # Quarantined URLs ride along: eligible() (Block 5) skips them instead
        # of rediscovering the failure. Router owns idx mapping; frontend owns
        # the inhibition table - the URL list is their shared key.
        inhibited = [
            u for u in list(getattr(self.router, "prefill_urls", []))
            + list(getattr(self.router, "decode_urls", []))
            if self.is_inhibited(u)
        ]
        try:
            route = self.router.route(token_ids, body.get("conversation_id"),
                                      inhibited_urls=inhibited)
        except RoutingError as e:
            # Filtering taxonomy: overload (compatible but all filtered) is a
            # capacity signal -> 503; nothing compatible at all -> 502.
            raise HTTPException(
                status_code=503 if e.reason == "overloaded" else 502,
                detail=f"no worker available ({e.reason})",
            )
        return await self.dispatch(route, body, token_ids, t_req,
                                   body.get("conversation_id"))

    # ── S4+S5 / S6-S9: DISPATCH ──────────────────────────────
    # Deviation from the diagram's sequential S5->S6, documented here and not
    # hidden: decode fires WITHOUT awaiting prefill. vLLM push-mode's
    # decode-side PUSH_REG is what UNBLOCKS the prefill's KV transfer, so
    # serializing (await prefill, then decode) holds every request ~2 s.
    # Same boxes as the diagram, overlapped in time.

    async def dispatch(self, route: Route, body: dict,
                       token_ids: list[int], t_req: float,
                       session_id: str | None = None):
        req_id = str(uuid.uuid4())
        headers = {"X-Request-Id": req_id}

        # S4/S5 producer contract lives in workers/prefill_worker.py (the box
        # that owns prefill semantics). See producer_body for the dual-cap WHY.
        p_body = producer_body(body)

        # S6: metadata injected by the router (its box owns the sub-label).
        d_body = self.router.inject_transfer_metadata(
            route, body, req_id, self.prefill_kv_host)

        log.info("PUSH[%s] P=%s D=%s engine=%s",
                 req_id[:8], route.prefill_url, route.decode_url, route.engine_id)
        p_task = asyncio.create_task(
            self._drain(f"http://{route.prefill_url}/v1/chat/completions",
                        p_body, headers)
        )
        try:
            # S8/S9 leg. Non-200 here cancels the producer: no decode means
            # no transfer consumer, so holding prefill open is pure waste.
            # No aggregated-path retry (official rule) -> 502 to the client.
            stream = await self.stream_decode(
                f"http://{route.decode_url}/v1/chat/completions", d_body, headers
            )
        except UpstreamError as e:
            p_task.cancel()
            # Inhibit the failed decode worker: the NEXT request's eligible()
            # (Block 5) skips it for INHIBIT_SECS instead of rediscovering
            # the same failure the hard way.
            self.inhibit(f"http://{route.decode_url}")
            log.error("Decode-side failure for PUSH[%s]: %s", req_id[:8], e)
            return JSONResponse(status_code=502, content={"error": str(e)})

        asyncio.create_task(self._finish_push(p_task, route, token_ids, req_id, t_req))

        # Completion tracking: the wrapped generator below fires
        # _complete_request when the client finishes consuming the stream.
        # Timing contract (official lifecycle, two halves):
        #   prefill side -> _finish_push (drain end == prefill+1 token done)
        #   decode side  -> here (stream exhaustion == generation done)
        async def tracked():
            completed = False
            try:
                async for chunk in stream:
                    yield chunk
                completed = True
            finally:
                self._complete_request(route, token_ids, session_id,
                                       completed=completed)

        return StreamingResponse(tracked(), media_type="text/event-stream")

    def _complete_request(self, route: Route, token_ids: list[int],
                          session_id: str | None, completed: bool) -> None:
        """Decode-side completion (sync: pure accounting, no I/O).
        Frees decode usage ALWAYS (load must not leak on disconnects);
        records decode blocks + binds affinity ONLY on full consumption
        (official commit-on-dispatch: partial streams don't pin sessions).
        All router access is getattr-guarded: stub routers in unit tests
        don't carry indexer/slots."""
        router = self.router
        if router is None:
            return
        slots = getattr(router, "slots", None)
        indexer = getattr(router, "indexer", None)
        if slots is None or indexer is None:
            return
        block_size = getattr(getattr(router, "kv", None), "block_size", 64)
        hashes = set(token_ids_to_block_hashes(token_ids, block_size))
        slots.free(route.p_idx, route.d_idx, len(hashes))
        if not completed:
            return
        indexer.record(list(hashes), route.d_idx, "decode")
        sessions = getattr(router, "sessions", None)
        if sessions is not None and session_id:
            sessions.bind(session_id, route, hashes)

    async def _finish_push(self, p_task, route: Route,
                           token_ids: list[int], req_id: str, t_req: float):
        """Post-S9: await the producer drain, then release + record.

        Load release + block recording land with slot_manager/kv_indexer
        (Block 5/6) - until then this only awaits and logs. Client stream is
        unaffected: it was returned before this task runs.
        """
        prefill_ok = True
        try:
            await p_task
        except Exception as e:  # producer failed; client already has decode stream
            prefill_ok = False
            log.error("PUSH[%s] prefill drain failed: %s", req_id[:8], e)
        # Prefill-side completion: drain end == prefill+1 token done (the
        # producer is capped), so this is the correct MarkPrefillCompleted
        # moment: release prompt charge + publish prefill blocks. Decode side
        # is handled by _complete_request at stream exhaustion.
        router = getattr(self, "router", None)
        slots = getattr(router, "slots", None)
        indexer = getattr(router, "indexer", None)
        if prefill_ok and slots is not None and indexer is not None:
            block_size = getattr(getattr(router, "kv", None), "block_size", 64)
            hashes = token_ids_to_block_hashes(token_ids, block_size)
            slots.mark_prefill_completed(route.p_idx, len(token_ids))
            indexer.record(hashes, route.p_idx, "prefill")
        log.info("PROF[%s] mode=push P%d D%d total_to_stream=%.1f isl=%d",
                 req_id[:8], route.p_idx, route.d_idx,
                 (time.monotonic() - t_req) * 1000, len(token_ids))

    # ── S8/S9: DECODE + RESPONSE ─────────────────────────────
    # response() (S9 badge, worker side) starts the stream; the bytes below
    # are the "Stream Tokens" edge relayed through this Frontend.

    async def stream_decode(self, url: str, body: dict, headers: dict):
        """S8/S9 relay: eager-POST decode worker, stream chunks back."""
        return await self._post_stream(url, body, headers)

    async def _drain(self, url: str, body: dict, headers: dict):
        """Prefill sink: same POST primitive, response consumed and discarded."""
        async for _ in await self._post_stream(url, body, headers):
            pass

    async def _post_stream(self, url: str, body: dict, headers: dict):
        """Single transport path. Returns an async chunk iterator; closes the
        connection when exhausted. Non-200 AND transport failures (refused,
        timeout, reset) raise UpstreamError (never None, never silent) -
        callers map it to status codes."""
        client = httpx.AsyncClient(timeout=HTTP_TIMEOUT)
        stream_ctx = client.stream("POST", url, json=body, headers=headers)
        try:
            resp = await stream_ctx.__aenter__()
        except (httpx.HTTPError, OSError) as e:
            await client.aclose()
            raise UpstreamError(f"Upstream {url} unreachable: {e}") from e
        if resp.status_code != 200:
            err = await resp.aread()
            await stream_ctx.__aexit__(None, None, None)
            await client.aclose()
            raise UpstreamError(
                f"Upstream {url} returned {resp.status_code}: "
                f"{err[:200].decode(errors='replace')}"
            )

        async def gen():
            try:
                async for chunk in resp.aiter_bytes():
                    yield chunk
            finally:
                await stream_ctx.__aexit__(None, None, None)
                await client.aclose()

        return gen()

    def http_response(self) -> None:
        """S9 return leg: the StreamingResponse built in dispatch()."""
        raise NotImplementedError("TODO Pass 4: reserved for post-processing hooks")


def create_app(model: str = "Qwen/Qwen3-14B-FP8", router_mode: str = "kv",
               discovery=None) -> FastAPI:
    """App factory (not a module-global app: tests build isolated instances).

    discovery=None (default): router unwired -> inference 501s, tests inject.
    discovery=<Discovery>: full live stack built here from the registry tables
    (router urls/engines/ports consistent by construction) + models source
    set to prefill worker 0 + SessionAffinity sharing the router's own
    indexer/slots (single state: affinity sees the same blocks scoring sees).
    """
    frontend = Frontend(model=model, router_mode=router_mode)
    if discovery is not None:
        from router.sessions import SessionAffinity

        frontend.router = discovery.build_router(mode=router_mode)
        frontend.router.sessions = SessionAffinity(
            frontend.router.indexer, frontend.router.slots
        )
        prefill = discovery.worker_discovery("prefill")
        if prefill:
            frontend.models_source_url = f"http://{prefill[0].url}"
    app = FastAPI(title="mini-dynamo")
    app.state.frontend = frontend

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models(raw_request: Request):
        # Passthrough to prefill worker 0 (it serves the model card; this
        # gateway holds configs for tokenizing, never weights). Unwired until
        # the workers layer sets models_source_url -> explicit 501, not a
        # connection-refused traceback.
        fe = raw_request.app.state.frontend
        if fe.models_source_url is None:
            raise HTTPException(status_code=501, detail="models source not wired yet")
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"{fe.models_source_url}/v1/models")
            return JSONResponse(content=r.json())

    @app.get("/status")
    async def status(raw_request: Request):
        fe = raw_request.app.state.frontend
        return {
            "model": fe.model,
            "router_mode": fe.router_mode,
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(raw_request: Request):
        body = await raw_request.json()
        return await raw_request.app.state.frontend.http_api_call(body)

    @app.post("/v1/completions")
    async def completions(raw_request: Request):
        body = await raw_request.json()
        return await raw_request.app.state.frontend.http_api_call(body)

    @app.get("/kvbm/status")
    async def kvbm_status(raw_request: Request):
        # Lightweight JSON observability. Official exposes Prometheus /metrics;
        # we deliberately add no prometheus_client dependency (see header).
        fe = raw_request.app.state.frontend
        slots = getattr(getattr(fe, "router", None), "slots", None)
        if slots is None:
            raise HTTPException(status_code=501, detail="router not wired yet")
        urls = getattr(fe.router, "decode_urls", [])
        return {
            "decode_usage": {
                url: slots.decode_usage(i) for i, url in enumerate(urls)
            },
            "router_mode": fe.router_mode,
        }

    return app


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    # Entry-point owns log config (library import must NOT configure logging:
    # without this, root stays WARNING and every log.info in this package -
    # PUSH lines, PROF lines, routing evidence - is silently dropped.
    # Found by GPU proof: /tmp/mini.log had zero PROF lines while serving
    # 200s, because nothing ever set root to INFO.)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="mini-dynamo Frontend (S1/S2/S9)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-14B-FP8")
    parser.add_argument("--http-port", type=int, default=8787)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument(
        "--router-mode", type=str, default="kv",
        help=f"one of {ROUTER_MODES} (other official modes fail fast)",
    )
    parser.add_argument("--prefill-ports", type=int, nargs="+", default=[8100],
                        help="prefill vLLM ports (one per P worker)")
    parser.add_argument("--decode-ports", type=int, nargs="+", default=[8200],
                        help="decode vLLM ports (one per D worker)")
    parser.add_argument("--prefill-side-ports", type=int, nargs="*", default=None,
                        help="NIXL side-channel ports for prefill workers "
                             "(default: 5600+i)")
    parser.add_argument("--prefill-kv-host", type=str, default="127.0.0.1",
                        help="side-channel host for S7 metadata")
    parser.add_argument("--block-size", type=int, default=64,
                        help="KV block size (must match vLLM --block-size)")
    args = parser.parse_args(argv)

    from infra.discovery import Discovery
    from workers.decode_worker import DecodeWorker
    from workers.prefill_worker import PrefillWorker

    side_ports = args.prefill_side_ports or [5600 + i for i in range(len(args.prefill_ports))]
    discovery = Discovery(
        prefill=[PrefillWorker(url=f"127.0.0.1:{p}", engine_id=f"prefill-{i}",
                               side_port=side_ports[i])
                 for i, p in enumerate(args.prefill_ports)],
        decode=[DecodeWorker(url=f"127.0.0.1:{p}", engine_id=f"decode-{i}")
                for i, p in enumerate(args.decode_ports)],
    )
    app = create_app(model=args.model, router_mode=args.router_mode,
                     discovery=discovery)
    app.state.frontend.prefill_kv_host = args.prefill_kv_host
    # Block size must agree with the engines': same tokens must hash to the
    # same blocks the workers actually cache, or overlap scoring is fiction.
    app.state.frontend.router.kv.block_size = args.block_size

    uvicorn.run(app, host=args.host, port=args.http_port)


if __name__ == "__main__":
    main()
