#!/usr/bin/env python3

"""Build a compact 2q/4q/8q complexity-scaling figure for QuantumPhysEval."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CATEGORY_ORDER = [
    "circuit_evolution",
    "measurement_prediction",
    "operator_composition",
    "entanglement_classification",
]

CATEGORY_LABELS = {
    "circuit_evolution": "Circuit evolution",
    "measurement_prediction": "Measurement prediction",
    "operator_composition": "Operator composition",
    "entanglement_classification": "Entanglement",
}

MODEL_LABELS = {
    "claude-sonnet-4-20250514": "Claude Sonnet 4",
    "claude-opus-4-1-20250805": "Claude Opus 4.1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a 2q/4q/8q complexity-scaling figure.")
    parser.add_argument("--results-2q", required=True)
    parser.add_argument("--results-4q", required=True)
    parser.add_argument("--results-8q", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-plot", required=True)
    return parser.parse_args()


def load_rows(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        rows = json.load(handle)
    return [row for row in rows if row.get("api_error") is None and row.get("d_phys") is not None]


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def summarize(rows: list[dict]) -> dict:
    overall = mean([float(row["d_phys"]) for row in rows])
    by_model: dict[str, list[float]] = defaultdict(list)
    by_category: dict[str, list[float]] = defaultdict(list)
    zero_count = 0
    full_count = 0
    for row in rows:
        value = float(row["d_phys"])
        by_model[str(row["model"])].append(value)
        by_category[str(row["category"])].append(value)
        if abs(value) < 1e-12:
            zero_count += 1
        if value >= 0.999999:
            full_count += 1
    return {
        "overall_mean_d_phys": overall,
        "zero_rate": zero_count / len(rows),
        "full_error_rate": full_count / len(rows),
        "per_model": {model: mean(values) for model, values in sorted(by_model.items())},
        "per_category": {category: mean(values) for category, values in sorted(by_category.items())},
        "count": len(rows),
    }


def apply_style(ax) -> None:
    ax.set_facecolor("#fbfaf7")
    ax.grid(True, axis="y", color="#ddd6cc", linewidth=0.9)
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", colors="#374151", labelsize=10)


def main() -> None:
    args = parse_args()
    rows_2q = load_rows(args.results_2q)
    rows_4q = load_rows(args.results_4q)
    rows_8q = load_rows(args.results_8q)

    summaries = {
        "2q": summarize(rows_2q),
        "4q": summarize(rows_4q),
        "8q": summarize(rows_8q),
    }
    summaries["deltas"] = {
        "overall_2q_to_4q": summaries["4q"]["overall_mean_d_phys"] - summaries["2q"]["overall_mean_d_phys"],
        "overall_4q_to_8q": summaries["8q"]["overall_mean_d_phys"] - summaries["4q"]["overall_mean_d_phys"],
        "overall_2q_to_8q": summaries["8q"]["overall_mean_d_phys"] - summaries["2q"]["overall_mean_d_phys"],
        "per_category_2q_to_8q": {
            category: summaries["8q"]["per_category"][category] - summaries["2q"]["per_category"][category]
            for category in CATEGORY_ORDER
        },
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)

    fig = plt.figure(figsize=(8.4, 8.2), facecolor="#fcfcfb")
    grid = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.15], hspace=0.42)
    ax_overall = fig.add_subplot(grid[0, 0])
    ax_category = fig.add_subplot(grid[1, 0])
    apply_style(ax_overall)
    apply_style(ax_category)

    x = np.array([2, 4, 8], dtype=float)
    qubit_labels = ["2 qubits", "4 qubits", "8 qubits"]
    colors = {
        "Claude Sonnet 4": "#c96f4a",
        "Claude Opus 4.1": "#2f6c8f",
        "Family mean": "#33425a",
    }
    line_specs = [
        ("Claude Sonnet 4", [summaries["2q"]["per_model"]["claude-sonnet-4-20250514"], summaries["4q"]["per_model"]["claude-sonnet-4-20250514"], summaries["8q"]["per_model"]["claude-sonnet-4-20250514"]]),
        ("Claude Opus 4.1", [summaries["2q"]["per_model"]["claude-opus-4-1-20250805"], summaries["4q"]["per_model"]["claude-opus-4-1-20250805"], summaries["8q"]["per_model"]["claude-opus-4-1-20250805"]]),
        ("Family mean", [summaries["2q"]["overall_mean_d_phys"], summaries["4q"]["overall_mean_d_phys"], summaries["8q"]["overall_mean_d_phys"]]),
    ]
    for label, values in line_specs:
        ax_overall.plot(
            x,
            values,
            marker="o",
            markersize=8,
            linewidth=2.5 if label != "Family mean" else 3.2,
            color=colors[label],
            label=label,
        )
        for xpos, ypos in zip(x, values):
            ax_overall.text(xpos, ypos + 0.022, f"{ypos:.3f}", ha="center", va="bottom", fontsize=9.2, color="#374151")
    ax_overall.set_xticks(x)
    ax_overall.set_xticklabels(qubit_labels)
    ax_overall.set_ylim(0.05, 0.72)
    ax_overall.set_ylabel("Mean normalized physics error")
    ax_overall.set_title(
        "A. Anthropic error rises steadily as the state space grows",
        loc="left",
        fontsize=12.5,
        fontweight="bold",
        pad=8,
    )
    ax_overall.legend(frameon=False, fontsize=9.4, loc="upper left")

    category_colors = {
        "circuit_evolution": "#b35d68",
        "measurement_prediction": "#d18a5a",
        "operator_composition": "#8295a8",
        "entanglement_classification": "#5f8d80",
    }
    for category in CATEGORY_ORDER:
        values = [
            summaries["2q"]["per_category"][category],
            summaries["4q"]["per_category"][category],
            summaries["8q"]["per_category"][category],
        ]
        ax_category.plot(
            x,
            values,
            marker="o",
            markersize=7.2,
            linewidth=2.2,
            color=category_colors[category],
            label=CATEGORY_LABELS[category],
        )
        ax_category.text(x[-1] + 0.2, values[-1], f"{values[-1]:.3f}", va="center", ha="left", fontsize=8.9, color="#374151")
    ax_category.set_xticks(x)
    ax_category.set_xticklabels(qubit_labels)
    ax_category.set_ylim(0.0, 1.02)
    ax_category.set_ylabel("Mean normalized physics error")
    ax_category.set_title(
        "B. The degradation is concentrated in exact quantum dynamics",
        loc="left",
        fontsize=12.5,
        fontweight="bold",
        pad=8,
    )
    ax_category.legend(frameon=False, fontsize=9.0, loc="upper left", ncol=2)

    fig.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.09)
    out_plot = Path(args.output_plot)
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_plot, dpi=240, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {out_json}")
    print(f"Wrote {out_plot}")


if __name__ == "__main__":
    main()
