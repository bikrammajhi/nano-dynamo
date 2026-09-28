"""Blue box: Prefill Worker handle. See docs/architecture.mmd (S4/S5).

A handle, not an implementation: prompt compute (compute_kv_cache) and the
transfer metadata (return_metadata/disaggregated_params) execute INSIDE vLLM.
What this gateway owns about prefill is the PRODUCER CONTRACT - the capped
request body that forces prefill+1-token semantics:

    producer_body(body): max_tokens=1 AND max_completion_tokens=1 (both:
    vLLM prefers the latter; one-field caps are silently ignored -> full OSL
    decode before transfer, the original 12.7x TTFT gap) + inert
    kv_transfer_params marking this leg producer-only.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PrefillWorker:
    """Static handle: where the worker is + how decode reaches it (S7 metadata)."""
    url: str            # base URL, e.g. "127.0.0.1:8100" (no scheme; dispatch adds http://)
    engine_id: str      # vLLM engine id, e.g. "prefill-0" (S6 remote_engine_id)
    side_port: int      # NIXL side-channel port, e.g. 5600 (S6 remote_port)
    gpu: int = 0        # informational only (placement documentation)
    # Prefill topology for S7 metadata: vLLM's push connector needs the
    # producer's tp/pp shape to plan the transfer. Single-GPU workers: 1/1.
    # (Old gateway hardcoded the same; a missing pair breaks decode
    # coordination -> 502. Found by GPU proof, not by reading.)
    tp_size: int = 1
    pp_size: int = 1

    @property
    def chat_url(self) -> str:
        return f"http://{self.url}/v1/chat/completions"

    @property
    def models_url(self) -> str:
        return f"http://{self.url}/v1/models"


def producer_body(body: dict) -> dict:
    """S4/S5 producer contract. Pure function of the client body (testable
    without workers): caps generation at 1 token so KV transfer starts after
    ~200 ms of prefill instead of ~2.5 s of full decode."""
    p_body = dict(body)
    p_body["max_tokens"] = 1
    p_body["max_completion_tokens"] = 1
    p_body["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }
    return p_body
