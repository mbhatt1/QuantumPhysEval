#!/usr/bin/env python3

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#b6aea3")
    ax.spines["bottom"].set_color("#b6aea3")
    ax.set_facecolor("#fbfaf7")
    ax.grid(axis="y", color="#ddd6cc", alpha=0.95, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#374151")


def draw_panel(ax, payload: dict, title: str):
    mode_order = ["no_retry", "self_retry", "verifier_feedback"]
    mode_labels = ["One-shot", "Self-retry", "Verifier\nfeedback"]
    colors = ["#9ca3af", "#2f6c8f", "#5f8d80"]
    values = [payload["overall"][mode]["solve_rate"] for mode in mode_order]
    xs = np.arange(len(values))

    style_axis(ax)
    bars = ax.bar(xs, values, color=colors, width=0.62, edgecolor="white", linewidth=1.2)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Solved fraction", color="#111827")
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_xticks(xs)
    ax.set_xticklabels(mode_labels)
    ax.set_title(title, loc="left", fontsize=17, fontweight="bold", color="#111827", pad=10)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.03,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=11.0,
            color="#1f2937",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.18", fc="#f6f2ec", ec="none", alpha=0.95),
        )
    ax.annotate(
        f"+{payload['deltas']['verifier_feedback_vs_no_retry']:.3f}",
        xy=(2, values[2] + 0.01),
        xytext=(0.9, min(0.95, values[2] + 0.16)),
        textcoords="data",
        ha="center",
        fontsize=11,
        color="#1f2937",
        arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#64748b"),
    )


def main() -> None:
    openai = load(ROOT / "artifacts" / "results" / "feedback_ablation_comparison.json")
    anthropic = load(ROOT / "artifacts" / "results" / "anthropic_feedback_ablation_comparison.json")
    out_path = ROOT / "artifacts" / "figures" / "feedback_ablation_summary.png"

    fig = plt.figure(figsize=(11.8, 6.0), facecolor="#fcfcfb")
    grid = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.24)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])

    draw_panel(
        ax_a,
        openai,
        "A. OpenAI repair ablation",
    )
    draw_panel(
        ax_b,
        anthropic,
        "B. Anthropic repair ablation",
    )

    fig.suptitle(
        "Direct repair result: retries help, verifier feedback helps more",
        x=0.06,
        y=0.995,
        ha="left",
        fontsize=13.5,
        color="#475569",
    )
    fig.text(
        0.06,
        0.94,
        "Both provider subsets show the same mode ordering: one-shot < self-retry < verifier feedback.",
        fontsize=10.6,
        color="#6b7280",
        ha="left",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
