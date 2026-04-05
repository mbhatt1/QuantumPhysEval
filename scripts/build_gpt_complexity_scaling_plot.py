#!/usr/bin/env python3

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

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

MODEL_ORDER = ["gpt-5.1", "gpt-4.1", "gpt-5.2", "gpt-5.4"]
MODEL_LABELS = {
    "gpt-5.1": "GPT-5.1",
    "gpt-4.1": "GPT-4.1",
    "gpt-5.2": "GPT-5.2",
    "gpt-5.4": "GPT-5.4",
}


def load_rows(path: Path, only_gpt: bool = False) -> list[dict]:
    rows = json.load(path.open("r", encoding="utf-8"))
    usable = [row for row in rows if row.get("api_error") is None and row.get("d_phys") is not None]
    if only_gpt:
        usable = [row for row in usable if str(row.get("model", "")).startswith("gpt-")]
    return usable


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def summarize(rows: list[dict]) -> dict:
    by_model: dict[str, list[float]] = defaultdict(list)
    by_category: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = float(row["d_phys"])
        by_model[str(row["model"])].append(value)
        by_category[str(row["category"])].append(value)
    return {
        "overall_mean_d_phys": mean([float(row["d_phys"]) for row in rows]),
        "per_model": {model: mean(values) for model, values in by_model.items()},
        "per_category": {category: mean(values) for category, values in by_category.items()},
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
    rows_2q = load_rows(ROOT / "artifacts" / "results" / "combined_quantum_results.json", only_gpt=True)
    rows_4q = load_rows(ROOT / "artifacts" / "results" / "gpt_4qubit_results.json", only_gpt=True)
    rows_8q = load_rows(ROOT / "artifacts" / "results" / "gpt_8qubit_results.json", only_gpt=True)

    summaries = {"2q": summarize(rows_2q), "4q": summarize(rows_4q), "8q": summarize(rows_8q)}
    out_json = ROOT / "artifacts" / "results" / "gpt_complexity_scaling_summary.json"
    out_plot = ROOT / "artifacts" / "figures" / "gpt_complexity_scaling.png"
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
    line_colors = {
        "GPT-5.1": "#6f7bb2",
        "GPT-4.1": "#2f6c8f",
        "GPT-5.2": "#5f8d80",
        "GPT-5.4": "#33425a",
        "Family mean": "#1f2937",
    }
    line_specs = [("Family mean", [summaries["2q"]["overall_mean_d_phys"], summaries["4q"]["overall_mean_d_phys"], summaries["8q"]["overall_mean_d_phys"]])]
    for model in MODEL_ORDER:
        if all(model in summaries[q]["per_model"] for q in ("2q", "4q", "8q")):
            line_specs.append((MODEL_LABELS[model], [summaries["2q"]["per_model"][model], summaries["4q"]["per_model"][model], summaries["8q"]["per_model"][model]]))
    for label, values in line_specs:
        ax_overall.plot(
            x,
            values,
            marker="o",
            markersize=8.0,
            linewidth=3.0 if label == "Family mean" else 2.2,
            color=line_colors[label],
            label=label,
        )
        ax_overall.text(
            x[-1] + 0.2,
            values[-1],
            f"{values[-1]:.3f}",
            va="center",
            ha="left",
            fontsize=8.9,
            color="#374151",
            bbox=dict(boxstyle="round,pad=0.18", fc="#f6f2ec", ec="none", alpha=0.95),
        )
    ax_overall.set_xticks(x)
    ax_overall.set_xticklabels(qubit_labels)
    ax_overall.set_ylim(0.30, 0.75)
    ax_overall.set_ylabel("Mean normalized physics error")
    ax_overall.set_title(
        "A. GPT error rises from 2 to 4 to 8 qubits",
        loc="left",
        fontsize=12.5,
        fontweight="bold",
        pad=8,
    )
    ax_overall.legend(frameon=False, fontsize=9.0, loc="upper left", ncol=2)

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
            markersize=7.0,
            linewidth=2.2,
            color=category_colors[category],
            label=CATEGORY_LABELS[category],
        )
        ax_category.text(
            x[-1] + 0.2,
            values[-1],
            f"{values[-1]:.3f}",
            va="center",
            ha="left",
            fontsize=8.9,
            color="#374151",
            bbox=dict(boxstyle="round,pad=0.18", fc="#f6f2ec", ec="none", alpha=0.95),
        )
    ax_category.set_xticks(x)
    ax_category.set_xticklabels(qubit_labels)
    ax_category.set_ylim(0.0, 1.02)
    ax_category.set_ylabel("Mean normalized physics error")
    ax_category.set_title(
        "B. GPT degradation is again concentrated in exact quantum dynamics",
        loc="left",
        fontsize=12.5,
        fontweight="bold",
        pad=8,
    )
    ax_category.legend(frameon=False, fontsize=9.0, loc="upper left", ncol=2)

    fig.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.09)
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_plot, dpi=240, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
