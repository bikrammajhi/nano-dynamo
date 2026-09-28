"""Naming contract: code symbols match docs/architecture.mmd node ids.

Rule: every symbol is official-literal, diagram-edge-derived, or cited
nano-local. Rename a concept -> update the .mmd first; this test fails until
both agree. Runs with no GPU, no network, no weights.
"""

import importlib


def _has(modname, clsname, fns):
    mod = importlib.import_module(modname)
    cls = getattr(mod, clsname)
    for fn in fns:
        assert callable(getattr(cls, fn)), f"{modname}.{clsname}.{fn}"
    return cls


def test_frontend_surface():
    # Orange box: S1/S2/S9 vocabulary from edge labels.
    _has("frontend.frontend", "Frontend", [
        "http_api_call", "preprocess", "tokenize_and_validate",
        "dispatch", "stream_decode", "http_response",
        "inhibit", "is_inhibited",
    ])


def test_prefill_router_surface():
    # Purple box: S3/S6 badges + sub-labels, verbatim.
    _has("router.prefill_router", "PrefillRouter", [
        "route", "route_to_prefill", "select_prefill_worker",
        "route_to_decode", "select_decode", "inject_transfer_metadata",
        "eligible",
    ])
    _has("router.prefill_router", "KvRouter", ["score"])
    _has("router.prefill_router", "Route", [])


def test_indexer_slots_events_surface():
    _has("router.kv_indexer", "KvIndexer", [
        "find_matches_for_request", "record", "forget", "overlap",
    ])
    _has("router.slot_manager", "SlotManager", [
        "add_request", "mark_prefill_completed", "free", "snapshot",
    ])
    mod = importlib.import_module("infra.kv_events")
    assert callable(mod.token_ids_to_block_hashes)


def test_sessions_workers_discovery_surface():
    # Dashed S3-entry note: hard-mode affinity verbs (bind/invalidate official).
    _has("router.sessions", "SessionAffinity", [
        "lookup", "bind", "invalidate", "access", "evict_lru",
    ])
    # Blue boxes: static handles (compute lives in vLLM, not here).
    # chat_url/models_url are properties (data, not actions) - asserted by
    # existence, not callability.
    import workers.prefill_worker as pw
    import workers.decode_worker as dw
    assert isinstance(getattr(pw.PrefillWorker, "chat_url"), property)
    assert isinstance(getattr(pw.PrefillWorker, "models_url"), property)
    assert callable(pw.producer_body)
    assert isinstance(getattr(dw.DecodeWorker, "chat_url"), property)
    # Yellow box: static registry + plane verbs.
    _has("infra.discovery", "Discovery", [
        "register_worker", "service_discovery", "worker_discovery",
        "build_router",
    ])


def test_no_router_layer_imports_cost():
    # Anti-overlap contract: frontend decides nothing (no scoring imports).
    import frontend.frontend as ff
    assert "KvRouter" not in dir(ff), "cost logic leaked into frontend"
    assert "cost_fn" not in dir(ff)


def test_no_planner_module():
    # Deletion decision enforced: a stub must break the build, not sneak back.
    try:
        importlib.import_module("planner")
    except ImportError:
        return
    raise AssertionError("nano_dynamo.planner must not exist (decision)")


def test_route_single_home():
    # Route owned by route()'s module, imported (not redefined) by frontend.
    import frontend.frontend as ff
    import router.prefill_router as pr
    assert ff.Route is pr.Route
