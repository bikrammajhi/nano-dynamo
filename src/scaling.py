"""Phase 5: Dynamic Pool Scaling

Manages worker lifecycle in the prefill/decode pools:
- Graceful drain (stop new requests, let existing finish)
- Pool rebalancing between prefill and decode roles
- Auto-scaling decisions based on utilization metrics
- Worker add/remove for external orchestrators
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Set, Tuple

log = logging.getLogger("nano-dynamo.scaling")


class ScalingManager:
    """Manages dynamic pool scaling for disaggregated inference.

    Maintains draining state per worker so the proxy can gracefully
    stop routing to a worker being removed. Provides metrics and
    recommendations for auto-scaling decisions.
    """

    def __init__(self):
        self._draining_prefill: Set[int] = set()
        self._draining_decode: Set[int] = set()
        self._scaling_events: List[dict] = []
        self._last_decision: float = 0.0

    def is_draining_prefill(self, idx: int) -> bool:
        return idx in self._draining_prefill

    def is_draining_decode(self, idx: int) -> bool:
        return idx in self._draining_decode

    def drain_prefill(self, idx: int) -> dict:
        self._draining_prefill.add(idx)
        event = {"action": "drain", "role": "prefill", "idx": idx,
                 "time": time.time()}
        self._scaling_events.append(event)
        log.info("SCALE drain prefill-%d", idx)
        return event

    def drain_decode(self, idx: int) -> dict:
        self._draining_decode.add(idx)
        event = {"action": "drain", "role": "decode", "idx": idx,
                 "time": time.time()}
        self._scaling_events.append(event)
        log.info("SCALE drain decode-%d", idx)
        return event

    def activate_prefill(self, idx: int) -> dict:
        self._draining_prefill.discard(idx)
        event = {"action": "activate", "role": "prefill", "idx": idx,
                 "time": time.time()}
        self._scaling_events.append(event)
        log.info("SCALE activate prefill-%d", idx)
        return event

    def activate_decode(self, idx: int) -> dict:
        self._draining_decode.discard(idx)
        event = {"action": "activate", "role": "decode", "idx": idx,
                 "time": time.time()}
        self._scaling_events.append(event)
        log.info("SCALE activate decode-%d", idx)
        return event

    def active_prefill_count(self, proxy) -> int:
        total = len(proxy.prefill_instances)
        return total - len(self._draining_prefill)

    def active_decode_count(self, proxy) -> int:
        total = len(proxy.decode_instances)
        return total - len(self._draining_decode)

    def pool_metrics(self, proxy) -> dict:
        """Return utilization metrics for auto-scaling decisions."""
        p_total = len(proxy.prefill_instances)
        d_total = len(proxy.decode_instances)
        p_active = self.active_prefill_count(proxy)
        d_active = self.active_decode_count(proxy)

        decode_loads = {
            f"D{i}": round(proxy.migration.decode_load(i), 3)
            for i in range(d_total)
        }
        overloaded = proxy.migration.overloaded_workers()

        return {
            "prefill": {"total": p_total, "active": p_active,
                        "draining": len(self._draining_prefill)},
            "decode": {"total": d_total, "active": d_active,
                       "draining": len(self._draining_decode)},
            "decode_loads": decode_loads,
            "overloaded": overloaded,
        }

    def auto_scale(self, proxy) -> Optional[dict]:
        """Evaluate load and recommend a scaling action.

        Returns scaling recommendation dict or None.
        This is advisory — an external orchestrator would act on it.
        """
        now = time.time()
        if now - self._last_decision < 10:
            return None

        metrics = self.pool_metrics(proxy)
        p_active = metrics["prefill"]["active"]
        d_active = metrics["decode"]["active"]

        overloaded = metrics["overloaded"]

        # Scale decode up if overloaded and we have spare prefill capacity
        if overloaded and p_active > 1 and d_active < len(proxy.decode_instances):
            self._last_decision = now
            rec = {
                "action": "rebalance",
                "reason": f"decode overloaded: {overloaded}",
                "from": f"{p_active}P+{d_active}D",
                "to": f"{p_active - 1}P+{d_active + 1}D",
            }
            log.info("SCALE recommend: %s", rec)
            return rec

        # Scale decode down if all are lightly loaded and decode > 1
        light_load = all(
            proxy.migration.decode_load(i) < 0.3
            for i in range(len(proxy.decode_instances))
        )
        if light_load and d_active > 1:
            self._last_decision = now
            rec = {
                "action": "down",
                "reason": "all decode workers lightly loaded",
                "from": f"{p_active}P+{d_active}D",
                "to": f"{p_active + 1}P+{d_active - 1}D",
            }
            log.info("SCALE recommend: %s", rec)
            return rec

        return None

    def events(self, limit: int = 20) -> List[dict]:
        return list(self._scaling_events[-limit:])
