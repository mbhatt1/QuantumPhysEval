#!/usr/bin/env python3

"""Build a summary figure for 2-qubit vs 4-qubit stress-test comparisons."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CATEGORY_ORDER = [
    "circuit_evolution",
    "measurement_prediction",
    "operator_composition",
    "entanglement_classification",
]

SHORT_CATEGORY_LABELS = {
    "circuit_evolution": "Circuit evolution",
    "measurement_prediction": "Measurement prediction",
    "operator_composition": "Operator composition",
    "entanglement_classification": "Entanglement",
}


def apply_plot_style(ax) -> None:
    ax.set_facecolor("white")
    ax.grid(True, axis="x", color="#e5e7eb", linewidth=0.8)
    ax.grid(False, axis="y")
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", colors="#374151", labelsize=10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pooled 2q vs 4q comparison artifacts.")
    parser.add_argument("--gpt-comparison", required=True)
    parser.add_argument("--anthropic-comparison", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-plot", required=True)
    return parser.parse_args()


def load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def category_delta(summary: dict, category: str) -> float:
    rows = [row for row in summary["per_model_category"] if row["category"] == category]
    return float(np.mean([float(row["delta_variant_minus_baseline"]) for row in rows]))


def main() -> None:
    args = parse_args()
    gpt = load(args.gpt_comparison)
    anthropic = load(args.anthropic_comparison)

    combined = {
        "families": [
            {
                "family": "GPT",
                **gpt["overall"],
            },
            {
                "family": "Anthropic",
                **anthropic["overall"],
            },
        ],
        "combined": {},
        "category_deltas": {},
    }
    combined["combined"] = {
        "baseline_mean_d_phys": float(
            np.average(
                [
                    gpt["overall"]["baseline_mean_d_phys"],
                    anthropic["overall"]["baseline_mean_d_phys"],
                ],
                weights=[gpt["baseline_count"], anthropic["baseline_count"]],
            )
        ),
        "variant_mean_d_phys": float(
            np.average(
                [
                    gpt["overall"]["variant_mean_d_phys"],
                    anthropic["overall"]["variant_mean_d_phys"],
                ],
                weights=[gpt["variant_count"], anthropic["variant_count"]],
            )
        ),
    }
    combined["combined"]["delta_variant_minus_baseline"] = (
        combined["combined"]["variant_mean_d_phys"] - combined["combined"]["baseline_mean_d_phys"]
    )
    combined["combined"]["baseline_count"] = int(gpt["baseline_count"] + anthropic["baseline_count"])
    combined["combined"]["variant_count"] = int(gpt["variant_count"] + anthropic["variant_count"])

    for category in CATEGORY_ORDER:
        combined["category_deltas"][category] = float(
            np.average(
                [
                    category_delta(gpt, category),
                    category_delta(anthropic, category),
                ],
                weights=[len(gpt["models"]), len(anthropic["models"])],
            )
        )

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as handle:
        json.dump(combined, handle, indent=2)

    fig = plt.figure(figsize=(8.0, 7.6), facecolor="white")
    grid = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.15], hspace=0.42)
    ax_family = fig.add_subplot(grid[0, 0])
    ax_category = fig.add_subplot(grid[1, 0])
    apply_plot_style(ax_family)
    apply_plot_style(ax_category)

    families = ["GPT", "Anthropic", "All models"]
    baseline = [
        gpt["overall"]["baseline_mean_d_phys"],
        anthropic["overall"]["baseline_mean_d_phys"],
        combined["combined"]["baseline_mean_d_phys"],
    ]
    variant = [
        gpt["overall"]["variant_mean_d_phys"],
        anthropic["overall"]["variant_mean_d_phys"],
        combined["combined"]["variant_mean_d_phys"],
    ]
    y = np.arange(len(families), dtype=float)
    for idx in range(len(families)):
        ax_family.plot(
            [baseline[idx], variant[idx]],
            [idx, idx],
            color="#d5dbe3",
            linewidth=3.2,
            solid_capstyle="round",
            zorder=1,
        )
    ax_family.scatter(
        baseline,
        y,
        s=96,
        color="#4f8fba",
        edgecolor="white",
        linewidth=1.0,
        label="2 qubits",
        zorder=2,
    )
    ax_family.scatter(
        variant,
        y,
        s=96,
        color="#c75c5c",
        edgecolor="white",
        linewidth=1.0,
        label="4 qubits",
        zorder=3,
    )
    ax_family.set_yticks(y)
    ax_family.set_yticklabels(families)
    ax_family.invert_yaxis()
    ax_family.set_xlim(0.0, max(variant) + 0.10)
    ax_family.set_xlabel("Mean normalized physics error")
    ax_family.set_title("A. 4-qubit stress test raises error across families", loc="left", fontsize=11.5, fontweight="bold", pad=8)
    for idx in range(len(families)):
        delta = variant[idx] - baseline[idx]
        ax_family.text(
            max(baseline[idx], variant[idx]) + 0.018,
            idx,
            f"+{delta:.3f}",
            va="center",
            ha="left",
            fontsize=9.0,
            color="#374151",
        )
    ax_family.legend(frameon=False, fontsize=9.2, loc="lower right")

    categories = list(CATEGORY_ORDER)
    deltas = [combined["category_deltas"][category] for category in categories]
    colors = [
        "#b56576" if category in {"circuit_evolution", "measurement_prediction"} else "#9fb3c2"
        for category in categories
    ]
    labels = [SHORT_CATEGORY_LABELS[category] for category in categories]
    ypos = np.arange(len(categories), dtype=float)
    ax_category.barh(ypos, deltas, color=colors, edgecolor="white", linewidth=1.0, height=0.56)
    ax_category.axvline(0.0, color="#8e99a8", linewidth=1.0)
    ax_category.set_yticks(ypos)
    ax_category.set_yticklabels(labels)
    ax_category.invert_yaxis()
    ax_category.set_xlabel("Mean error delta (4q - 2q)")
    ax_category.set_title("B. The degradation is concentrated in dynamics-heavy tasks", loc="left", fontsize=11.5, fontweight="bold", pad=8)
    ax_category.set_xlim(min(-0.05, min(deltas) - 0.03), max(deltas) + 0.06)
    for ypos_value, value in zip(ypos, deltas):
        ax_category.text(
            value + (0.008 if value >= 0 else -0.008),
            ypos_value,
            f"{value:+.3f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=9.0,
            color="#374151",
        )

    out_plot = Path(args.output_plot)
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.18, right=0.96, top=0.94, bottom=0.10)
    fig.savefig(out_plot, dpi=240, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {out_json}")
    print(f"Wrote {out_plot}")


if __name__ == "__main__":
    main()
