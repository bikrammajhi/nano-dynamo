"""Generate the head-to-head benchmark chart (Nano-Dynamo vs NVIDIA Dynamo, 2P+2D).

Data source: BENCHMARK_RESULTS.md, 2026-08-01 runs (Qwen3-14B-FP8, 4x A100,
AIPerf, push mode). Usage: python benchmark_charts.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCENARIOS = ["multi_turn", "mixed_workload"]
NANO = {
    "multi_turn": {"ttft": 264, "tok_s": 371, "lat": 2083},
    "mixed_workload": {"ttft": 326, "tok_s": 1121, "lat": 3085},
}
DYNAMO = {
    "multi_turn": {"ttft": 195, "tok_s": 405, "lat": 1992},
    "mixed_workload": {"ttft": 247, "tok_s": 1155, "lat": 2984},
}

NANO_COLOR = "#4C72B0"
DYNAMO_COLOR = "#DD8452"
OUT = "docs/benchmark_2p2d_qwen3_14b.png"


def draw(ax, key, ylabel, title, annotate_gap=False):
    n = [NANO[s][key] for s in SCENARIOS]
    d = [DYNAMO[s][key] for s in SCENARIOS]
    x = np.arange(len(SCENARIOS))
    width = 0.36
    ax.bar(x - width / 2, n, width, label="Nano-Dynamo", color=NANO_COLOR)
    ax.bar(x + width / 2, d, width, label="NVIDIA Dynamo", color=DYNAMO_COLOR)
    if annotate_gap:
        for i, (a, b) in enumerate(zip(n, d)):
            ax.annotate(
                f"{a / b:.2f}x", (i + width / 2, max(a, b)),
                textcoords="offset points", xytext=(0, 4),
                ha="center", fontsize=10, fontweight="bold",
            )
    ax.set_xticks(x)
    ax.set_xticklabels(SCENARIOS, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)


fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
draw(axes[0], "ttft", "TTFT (ms)", "Time to first token — lower is better", annotate_gap=True)
draw(axes[1], "tok_s", "Throughput (tok/s)", "Throughput — higher is better", annotate_gap=True)
draw(axes[2], "lat", "Request latency (ms)", "End-to-end latency — lower is better", annotate_gap=True)

fig.legend(
    loc="upper center", ncol=2, frameon=False, fontsize=11,
    bbox_to_anchor=(0.5, 1.0),
)
fig.suptitle(
    "Nano-Dynamo vs NVIDIA Dynamo — 2P+2D, Qwen3-14B-FP8, AIPerf (gap = Nano / Dynamo)",
    y=1.06, fontsize=12,
)
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"wrote {OUT}")
