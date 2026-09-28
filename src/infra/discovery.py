"""Yellow box (dotted): Discovery. See docs/architecture.mmd.

Dev-mode extreme of the discovery plane: static membership instead of
etcd/K8s/memory backends. Registration, watches, and leases don't exist;
the tables below ARE the registry. Consequences, stated plainly:
  - editing membership = editing this file (or passing custom tables);
  - restart = amnesia (no worker state survives; indexer/slots are rebuilt
    from serving traffic, never from disk);
  - a wrong port fails at first request, not at startup (no health gate).

Official counterpart: lib/runtime discovery plane. When a real backend ever
lands, it implements this same interface (register_worker /
service_discovery / worker_discovery) and these tables become its seed.
"""

from __future__ import annotations

import logging
from typing import Dict, List

from workers.decode_worker import DecodeWorker
from workers.prefill_worker import PrefillWorker

log = logging.getLogger("mini-dynamo.discovery")


def default_prefill_workers() -> List[PrefillWorker]:
    """2P layout matching the load harness (modal_bench.py):
    GPU ids are placement documentation; side ports are S7 metadata."""
    return [
        PrefillWorker(url="127.0.0.1:8100", engine_id="prefill-0",
                      side_port=5600, gpu=0),
        PrefillWorker(url="127.0.0.1:8101", engine_id="prefill-1",
                      side_port=5601, gpu=1),
    ]


def default_decode_workers() -> List[DecodeWorker]:
    return [
        DecodeWorker(url="127.0.0.1:8200", engine_id="decode-0", gpu=2),
        DecodeWorker(url="127.0.0.1:8201", engine_id="decode-1", gpu=3),
    ]


class Discovery:
    """Static registry. Sole home of membership (anti-overlap rule #4):
    PrefillRouter receives these tables as constructor args, never copies."""

    def __init__(
        self,
        prefill: List[PrefillWorker] | None = None,
        decode: List[DecodeWorker] | None = None,
    ):
        self.prefill = prefill if prefill is not None else default_prefill_workers()
        self.decode = decode if decode is not None else default_decode_workers()

    def register_worker(self, worker) -> None:
        """Static-mode registration = admission log only. There is no watch
        to notify and no lease to grant; the tables are the truth."""
        log.info("REGISTER static worker=%s (no-op: tables are the registry)",
                 getattr(worker, "url", worker))

    def service_discovery(self) -> Dict[str, List[str]]:
        """What the Frontend dotted edge resolves: all known base URLs."""
        return {
            "prefill": [w.url for w in self.prefill],
            "decode": [w.url for w in self.decode],
        }

    def worker_discovery(self, role: str) -> list:
        """What the PrefillRouter dotted edge resolves: handles per role."""
        if role == "prefill":
            return list(self.prefill)
        if role == "decode":
            return list(self.decode)
        raise ValueError(f"unknown role {role!r} (want 'prefill'|'decode')")

    def build_router(self, **kwargs):
        """One-call wiring: PrefillRouter fed from these tables (urls, engine
        ids, side ports stay consistent by construction - no second copy)."""
        from router.prefill_router import PrefillRouter

        return PrefillRouter(
            prefill_urls=[w.url for w in self.prefill],
            decode_urls=[w.url for w in self.decode],
            engine_ids=[w.engine_id for w in self.prefill],
            prefill_side_ports=[w.side_port for w in self.prefill],
            tp_size=self.prefill[0].tp_size if self.prefill else 1,
            pp_size=self.prefill[0].pp_size if self.prefill else 1,
            **kwargs,
        )
