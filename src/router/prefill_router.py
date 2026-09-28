"""Purple box: PrefillRouter (S3 + S6). See docs/architecture.mmd.

Owns worker selection and NOTHING else: no HTTP, no bytes, no tokenizer.
Filter -> score -> pick pipeline (cf. Router Filtering doc):
  route()             S3 then S6; affinity short-circuit first (sessions.py, Block 6)
  eligible()          hard filter: inhibited URLs + busy thresholds (default off)
  select_prefill_worker()  S3: KvRouter cost scoring (overlap + load)
  select_decode()     S6: load-only (official decode-leg flags)

Cost model (Routing Concepts doc, term-for-term):
  raw_prefill      = (active_prefill_tokens + ISL) / block_size
  overlap_credit   = effective_credit * overlap_blocks      # device tier only;
                     host/disk/shared tiers deliberately absent (comment below)
  adjusted         = max(raw_prefill - overlap_credit, 0)   # clamped, never negative
  potential_decode = active_decode_blocks + incoming_blocks # incoming counted:
                     the request's own blocks WILL occupy decode VRAM
  request_cost     = decode_active_request_weight * active_requests
  logit            = prefill_load_scale * adjusted + potential_decode + request_cost
  -> argmin (temp=0, reservoir tie-break) / range-normalized softmin (temp>0)

Official counterpart for scoring: lib/kv-router (protocols.rs core,
scheduling/ policies, conditional_disagg.rs for the S6 exception).
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

from infra.kv_events import token_ids_to_block_hashes
from router.kv_indexer import KvIndexer
from router.slot_manager import SlotManager

log = logging.getLogger("mini-dynamo.router")

ROUTER_MODES = ("round-robin", "random", "kv")


class RoutingError(RuntimeError):
    """No worker could take the request. reason: 'empty' (nothing compatible -
    maps to 502) or 'overloaded' (compatible exists but all filtered -
    maps to 503). Mirrors the Filtering doc's error taxonomy."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass
class Route:
    """S3+S6 answer: where this request goes and how decode reaches prefill.
    Owned here (it is route()'s return). Frontend imports it; never defines it."""
    prefill_url: str
    decode_url: str
    p_idx: int
    engine_id: str
    prefill_side_port: int
    d_idx: int
    tp_size: int = 1   # producer topology for S7 metadata (see PrefillWorker)
    pp_size: int = 1


@dataclass
class KvRouter:
    """The scoring math (S3 dashed zoom-in: Cost fn). Pure: numbers in, logit out."""

    overlap_score_credit: float = 1.0
    overlap_score_credit_decay: float = 0.0
    prefill_load_scale: float = 1.0
    router_temperature: float = 0.0
    decode_active_request_weight: float = 0.0
    block_size: int = 64

    def score(self, *, isl_tokens: int, overlap_blocks: int,
              active_prefill_tokens: int, active_decode_blocks: int,
              active_requests: int, overlap_credit: Optional[float] = None,
              least_prefill_blocks: float = 0.0) -> float:
        """One worker's logit. overlap_credit=None -> use configured credit
        (S6 passes 0.0: official decode-leg exception, no prefix chasing)."""
        bs = float(self.block_size)
        raw_prefill = (active_prefill_tokens + isl_tokens) / bs
        credit = self.overlap_score_credit if overlap_credit is None else overlap_credit
        if self.overlap_score_credit_decay > 0 and least_prefill_blocks > 0:
            # Decay device credit on overloaded workers (Tuning doc):
            # 1 / (1 + decay * normalized_excess). Host/disk/shared tiers
            # don't exist here, so only the device term decays.
            excess = max(active_prefill_tokens / bs - least_prefill_blocks, 0.0)
            request_blocks = max(int(math.ceil(isl_tokens / self.block_size)), 1)
            credit *= 1.0 / (1.0 + self.overlap_score_credit_decay
                             * (excess / request_blocks))
        adjusted = max(raw_prefill - credit * overlap_blocks, 0.0)
        potential_decode = active_decode_blocks  # caller adds incoming blocks
        request_cost = self.decode_active_request_weight * active_requests
        return self.prefill_load_scale * adjusted + potential_decode + request_cost


class PrefillRouter:
    """Purple box. Decides (S3+S6); executes nothing."""

    def __init__(
        self,
        *,
        prefill_urls: List[str],
        decode_urls: List[str],
        engine_ids: Optional[List[str]] = None,
        prefill_side_ports: Optional[List[int]] = None,
        indexer: Optional[KvIndexer] = None,
        slots: Optional[SlotManager] = None,
        sessions=None,  # sessions.py lands in Block 6; None = skip affinity
        mode: str = "kv",
        kv: Optional[KvRouter] = None,
        # Busy thresholds (Filtering doc). None = off (official opt-in posture;
        # off by default so benchmarks measure scoring, not filtering).
        active_prefill_tokens_threshold: Optional[int] = None,
        decode_blocks_threshold: Optional[int] = None,
        tp_size: int = 1,  # prefill topology: threaded into Route for S7 metadata
        pp_size: int = 1,
    ):
        if mode not in ROUTER_MODES:
            raise NotImplementedError(
                f"--router-mode={mode!r} not implemented "
                f"(official modes power-of-two/least-loaded/direct/"
                f"device-aware-weighted need their own policies). "
                f"Supported: {ROUTER_MODES}."
            )
        self.prefill_urls = list(prefill_urls)
        self.decode_urls = list(decode_urls)
        self.engine_ids = engine_ids or [f"prefill-{i}" for i in range(len(prefill_urls))]
        self.side_ports = prefill_side_ports or [5600 + i for i in range(len(prefill_urls))]
        self.indexer = indexer or KvIndexer()
        self.slots = slots or SlotManager()
        self.sessions = sessions
        self.mode = mode
        self.kv = kv or KvRouter()
        self.prefill_threshold = active_prefill_tokens_threshold
        self.decode_threshold = decode_blocks_threshold
        self.tp_size = tp_size
        self.pp_size = pp_size
        self._rr = 0

    # ── entry ────────────────────────────────────────────────
    def route(self, token_ids: List[int], session_id: Optional[str] = None,
              inhibited_urls: Sequence[str] = (),
              overlap_credit: Optional[float] = None) -> Route:
        """S3 then S6. overlap_credit: per-request direct-API/EPP parity
        (official blesses router_config_override.overlap_score_credit;
        scale/temperature stay constructor-level - official has no
        per-request form for those)."""
        if self.sessions is not None and session_id:
            bound = self.sessions.lookup(session_id)  # Block 6 activates
            if bound is not None:
                return bound

        if self.mode == "round-robin":
            p_idx = self._rr % len(self.prefill_urls)
            self._rr += 1
            d_idx = self._rr % len(self.decode_urls)
            return self._make_route(p_idx, d_idx)
        if self.mode == "random":
            return self._make_route(
                random.randrange(len(self.prefill_urls)),
                random.randrange(len(self.decode_urls)),
            )

        # kv mode: filter -> score -> pick, per stage.
        hashes = set(token_ids_to_block_hashes(token_ids, self.kv.block_size))
        p_idx = self.route_to_prefill(token_ids, hashes, inhibited_urls,
                                      overlap_credit=overlap_credit)
        d_idx = self.route_to_decode(hashes, inhibited_urls)

        self.slots.add_request(p_idx, d_idx, len(token_ids), len(hashes))
        route = self._make_route(p_idx, d_idx)
        # NOTE: binding happens in frontend._complete_request (it owns the
        # hashes + knows the stream actually finished). route() only READS
        # affinity (lookup above), never commits - a failed route must not
        # pin a session (official commit-on-dispatch rule).
        log.info("KVBM route P%d D%d overlap=%d isl=%d",
                 p_idx, d_idx,
                 self.indexer.overlap(p_idx, "prefill", hashes), len(token_ids))
        return route

    def route_to_prefill(self, token_ids: List[int], hashes: Set[int],
                           inhibited_urls: Sequence[str] = (),
                           overlap_credit: Optional[float] = None) -> int:
        """S3 badge: filter prefill pool, score survivors, return winner."""
        cands = self.eligible("prefill", list(range(len(self.prefill_urls))),
                              inhibited_urls)
        if not cands:
            raise RoutingError("overloaded"
                               if self._any_compatible("prefill", inhibited_urls)
                               else "empty")
        return self.select_prefill_worker(token_ids, hashes, cands,
                                          overlap_credit=overlap_credit)

    def route_to_decode(self, hashes: Set[int],
                        inhibited_urls: Sequence[str] = ()) -> int:
        """S6 badge: filter decode pool, load-pick winner."""
        cands = self.eligible("decode", list(range(len(self.decode_urls))),
                              inhibited_urls)
        if not cands:
            raise RoutingError("overloaded"
                               if self._any_compatible("decode", inhibited_urls)
                               else "empty")
        return self.select_decode(hashes, cands)

    def inject_transfer_metadata(self, route: Route, body: dict,
                                 req_id: str, kv_host: str) -> dict:
        """S6 sub-label: staple prefill transfer metadata onto the decode
        request (our `disaggregated_params`, vLLM block-ID flavor). Router
        passes paper, never bytes - NIXL moves them (S7, no code)."""
        d_body = dict(body)
        d_body["kv_transfer_params"] = {
            "do_remote_decode": False,
            "do_remote_prefill": True,
            "remote_engine_id": route.engine_id,
            "remote_host": kv_host,
            "remote_port": route.prefill_side_port,
            "tp_size": route.tp_size,   # producer topology: required by the
            "pp_size": route.pp_size,   # push connector's transfer planner
            "remote_request_id": req_id,
        }
        return d_body

    def _make_route(self, p_idx: int, d_idx: int) -> Route:
        return Route(
            prefill_url=self.prefill_urls[p_idx],
            decode_url=self.decode_urls[d_idx],
            p_idx=p_idx,
            engine_id=self.engine_ids[p_idx],
            prefill_side_port=self.side_ports[p_idx],
            d_idx=d_idx,
            tp_size=self.tp_size,
            pp_size=self.pp_size,
        )

    def _any_compatible(self, role: str, inhibited_urls: Sequence[str]) -> bool:
        """Any worker existing at all? Inhibition/busy are unavailability of
        existing capacity (-> overloaded/503); only a truly empty pool is
        'empty' (-> 502, misconfiguration)."""
        urls = self.prefill_urls if role == "prefill" else self.decode_urls
        return len(urls) > 0

    # ── filter stage (Router Filtering) ──────────────────────
    def eligible(self, role: str, candidates: List[int],
                 inhibited_urls: Sequence[str] = ()) -> List[int]:
        """Hard filter BEFORE scoring. Two rules (official subset):
        inhibited URLs (5 s quarantine from the frontend 502 path) and busy
        thresholds (None = off). Returns survivors; empty is a signal, not
        an error - route() classifies it (empty vs overloaded)."""
        urls = self.prefill_urls if role == "prefill" else self.decode_urls
        out = []
        for i in candidates:
            if urls[i] in inhibited_urls:
                continue
            if role == "prefill" and self.prefill_threshold is not None:
                if self.slots.active_prefill_tokens(i) > self.prefill_threshold:
                    continue
            if role == "decode" and self.decode_threshold is not None:
                if self.slots.decode_usage(i) > self.decode_threshold:
                    continue
            out.append(i)
        return out

    # ── S3: ROUTE TO PREFILL ─────────────────────────────────
    def select_prefill_worker(self, token_ids: List[int], hashes: Set[int],
                              candidates: List[int],
                              overlap_credit: Optional[float] = None) -> int:
        scores = {w: self.indexer.overlap(w, "prefill", hashes) for w in candidates}
        loads = {w: self.slots.snapshot(w) for w in candidates}
        least = 0.0
        if self.kv.overlap_score_credit_decay > 0:
            least = min(loads[w].active_prefill_tokens / float(self.kv.block_size)
                        for w in candidates)
        logits = {
            w: self.kv.score(
                isl_tokens=len(token_ids), overlap_blocks=scores[w],
                active_prefill_tokens=loads[w].active_prefill_tokens,
                active_decode_blocks=loads[w].active_decode_blocks,
                active_requests=loads[w].active_requests,
                overlap_credit=overlap_credit, least_prefill_blocks=least,
            )
            for w in candidates
        }
        chosen = self._pick(logits)
        log.info("KV_ROUTE[cost] worker=%d logit=%.3f overlap_blocks=%d isl=%d",
                 chosen, logits[chosen], scores[chosen], len(token_ids))
        return chosen

    # ── S6: ROUTE TO DECODE ──────────────────────────────────
    # Official decode-leg flags: overlap_score_credit=0 (no prefix chasing),
    # assume_kv_reuse=false, track_prefill_tokens=false. Decode load is decode
    # work only: usage + this request's incoming blocks, reservoir ties.
    def select_decode(self, hashes: Set[int], candidates: List[int]) -> int:
        incoming = len(hashes)
        logits = {d: max(self.slots.decode_usage(d) + incoming, 0)
                  for d in candidates}
        return self._pick(logits)

    # ── picking ──────────────────────────────────────────────
    def _pick(self, logits: Dict[int, float]) -> int:
        temp = self.kv.router_temperature
        if temp > 0:
            return self._softmin(logits, temp)
        best, best_logit, ties = None, float("inf"), 0
        for wid, logit in logits.items():  # reservoir: unbiased ties (Dynamo-equiv)
            if logit < best_logit:
                best, best_logit, ties = wid, logit, 1
            elif logit == best_logit:
                ties += 1
                if random.randint(0, ties - 1) == 0:
                    best = wid
        return best

    def _softmin(self, logits: Dict[int, float], temp: float) -> int:
        values = list(logits.values())
        lo, hi = min(values), max(values)
        if lo == hi:
            return random.choice(list(logits.keys()))
        scale = -1.0 / ((hi - lo) * temp)  # range-normalized (Dynamo-equiv)
        ceiling = lo * scale
        exps = [math.exp(v * scale - ceiling) for v in values]
        total = sum(exps)
        r, cumulative = random.random(), 0.0
        for wid, e in zip(logits.keys(), exps):
            cumulative += e / total
            if r <= cumulative:
                return wid
        return list(logits.keys())[-1]
