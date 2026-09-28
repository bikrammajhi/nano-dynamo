"""Blue box: Decode Worker handle. See docs/architecture.mmd (S6/S7/S8/S9).

Same handle-not-implementation rule as prefill_worker.py. The S-labelled
verbs execute in two places, stated explicitly so nothing reads as ours
when it is vLLM's:
  - kv_transfer (S7), decode/compute (S8): INSIDE vLLM via NixlPushConnector.
    This gateway supplies endpoints + metadata and never touches bytes.
  - response/stream start (S9 badge): vLLM emits SSE; frontend.stream_decode
    relays it (the "Stream Tokens" edge).

What this module owns: the static handle + the decode-leg selection note
(load-only; the scoring itself lives in PrefillRouter.select_decode per the
official decode-leg flags).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DecodeWorker:
    """Static handle: where the worker is."""
    url: str            # base URL, e.g. "127.0.0.1:8200"
    engine_id: str = ""  # informational (decode engines need no remote id)
    gpu: int = 2        # informational only (placement documentation)

    @property
    def chat_url(self) -> str:
        return f"http://{self.url}/v1/chat/completions"
