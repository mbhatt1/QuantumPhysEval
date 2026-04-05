#!/usr/bin/env python3

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "artifacts" / "figures" / "quantumphys_eval_overview.png"


def main() -> None:
    palette = {
        "bg": "#fcfcfb",
        "ink": "#16202a",
        "muted": "#5f6b76",
        "line": "#cfc8bd",
        "blue": "#2f6c8f",
        "blue_soft": "#dce9f0",
        "coral": "#c96f4a",
        "coral_soft": "#f1ddd1",
        "teal": "#5f8d80",
        "teal_soft": "#dce8e2",
        "navy": "#33425a",
        "navy_soft": "#dde4ec",
    }

    fig, ax = plt.subplots(figsize=(14.4, 6.1), facecolor=palette["bg"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.03, 0.93, "QuantumPhysEval pipeline", fontsize=20, fontweight="bold", color=palette["ink"])
    ax.text(
        0.03,
        0.875,
        "The benchmark is self-labeled: every prompt has exact quantum ground truth, so model error is measured directly rather than judged heuristically.",
        fontsize=11,
        color=palette["muted"],
    )

    stages = [
        ((0.03, 0.33), (0.21, 0.43), "1. Construct tasks", ["Exact circuit, operator,", "measurement, and", "entanglement prompts"], palette["blue_soft"], palette["blue"]),
        ((0.28, 0.33), (0.21, 0.43), "2. Compute targets", ["Analytic state vectors", "Exact unitaries and", "Born-rule probabilities"], palette["navy_soft"], palette["navy"]),
        ((0.53, 0.33), (0.21, 0.43), "3. Evaluate models", ["GPT and Claude families", "Depth settings 1, 2, 4", "Machine-readable outputs"], palette["coral_soft"], palette["coral"]),
        ((0.78, 0.33), (0.19, 0.43), "4. Score + analyze", [r"$d_{phys}\in[0,1]$", "Scaling fits, rankings,", "stress tests, repairs"], palette["teal_soft"], palette["teal"]),
    ]

    for (x, y), (w, h), title, body, facecolor, accent in stages:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.012,rounding_size=0.025",
                linewidth=1.2,
                edgecolor=palette["line"],
                facecolor=facecolor,
            )
        )
        ax.add_patch(
            FancyBboxPatch(
                (x + 0.016, y + h - 0.10),
                0.085,
                0.055,
                boxstyle="round,pad=0.01,rounding_size=0.025",
                linewidth=0,
                facecolor=accent,
                alpha=1.0,
            )
        )
        ax.text(x + 0.0585, y + h - 0.072, title.split(".")[0], fontsize=8.6, fontweight="bold", color="white", ha="center", va="center")
        ax.text(x + 0.02, y + h - 0.142, title, fontsize=12, fontweight="bold", va="top", color=palette["ink"])
        ax.text(x + 0.02, y + h - 0.25, "\n".join(body), fontsize=9.8, va="top", color=palette["muted"], linespacing=1.46)

    for x0, x1 in [(0.235, 0.28), (0.485, 0.53), (0.73, 0.77)]:
        ax.add_patch(
            FancyArrowPatch((x0, 0.545), (x1, 0.545), arrowstyle="-|>", mutation_scale=16, linewidth=1.8, color="#8d989f")
        )

    ax.add_patch(
        FancyBboxPatch(
            (0.03, 0.08),
            0.94,
            0.17,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            linewidth=1.2,
            edgecolor=palette["line"],
            facecolor="#f8f5ef",
        )
    )
    ax.text(0.05, 0.215, "Outputs used in the paper", fontsize=11.6, fontweight="bold", color=palette["ink"], va="center")
    chips = [
        ("Pooled 2q benchmark", palette["blue"]),
        ("Matched 4q stress test", palette["coral"]),
        ("8q complexity ladders", palette["navy"]),
        ("Verifier-feedback ablations", palette["teal"]),
    ]
    chip_x = 0.24
    gap = 0.02
    widths = [0.10 + 0.0026 * len(label) for label, _ in chips]
    for (label, color), width in zip(chips, widths):
        ax.add_patch(
            FancyBboxPatch(
                (chip_x, 0.152),
                width,
                0.06,
                boxstyle="round,pad=0.01,rounding_size=0.022",
                linewidth=0,
                facecolor=color,
                alpha=0.96,
            )
        )
        ax.text(chip_x + width / 2, 0.182, label, fontsize=8.4, color="white", ha="center", va="center", fontweight="bold")
        chip_x += width + gap

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=240, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
