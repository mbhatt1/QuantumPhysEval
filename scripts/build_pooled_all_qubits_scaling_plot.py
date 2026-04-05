#!/usr/bin/env python3

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_rows(path: Path) -> list[dict]:
    rows = json.loads(path.read_text())
    return [row for row in rows if row.get("api_error") is None and row.get("d_phys") is not None]


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def summarize(rows: list[dict]) -> dict:
    provider_groups: dict[str, list[float]] = {"GPT": [], "Claude": []}
    category_groups: dict[str, list[float]] = {
        "circuit_evolution": [],
        "measurement_prediction": [],
        "operator_composition": [],
        "entanglement_classification": [],
    }
    for row in rows:
        value = float(row["d_phys"])
        model = str(row["model"])
        provider_groups["GPT" if model.startswith("gpt-") else "Claude"].append(value)
        category_groups[str(row["category"])].append(value)
    return {
        "overall": mean([float(row["d_phys"]) for row in rows]),
        "provider": {name: mean(vals) for name, vals in provider_groups.items()},
        "category": {name: mean(vals) for name, vals in category_groups.items()},
        "count": len(rows),
    }


def style(ax) -> None:
    ax.set_facecolor("#fbfaf7")
    ax.grid(True, axis="y", color="#ddd6cc", linewidth=0.9)
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", colors="#374151", labelsize=10)


def main() -> None:
    rows_2q = load_rows(ROOT / "artifacts" / "results" / "combined_quantum_results.json")
    rows_4q = load_rows(ROOT / "artifacts" / "results" / "combined_4qubit_results.json")
    rows_8q = load_rows(ROOT / "artifacts" / "results" / "combined_8qubit_results.json")

    s2, s4, s8 = summarize(rows_2q), summarize(rows_4q), summarize(rows_8q)
    payload = {"2q": s2, "4q": s4, "8q": s8}
    (ROOT / "artifacts" / "results" / "combined_all_qubits_scaling_summary.json").write_text(
        json.dumps(payload, indent=2)
    )

    fig = plt.figure(figsize=(8.5, 8.3), facecolor="#fcfcfb")
    grid = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.15], hspace=0.42)
    ax_top = fig.add_subplot(grid[0, 0])
    ax_bottom = fig.add_subplot(grid[1, 0])
    style(ax_top)
    style(ax_bottom)

    x = np.array([2, 4, 8], dtype=float)
    xlabels = ["2 qubits", "4 qubits", "8 qubits"]

    top_series = [
        ("Pooled mean", [s2["overall"], s4["overall"], s8["overall"]], "#33425a", 3.0),
        ("GPT family", [s2["provider"]["GPT"], s4["provider"]["GPT"], s8["provider"]["GPT"]], "#2f6c8f", 2.2),
        ("Claude family", [s2["provider"]["Claude"], s4["provider"]["Claude"], s8["provider"]["Claude"]], "#c96f4a", 2.2),
    ]
    for label, values, color, width in top_series:
        ax_top.plot(x, values, marker="o", markersize=8, linewidth=width, color=color, label=label)
        ax_top.text(x[-1] + 0.18, values[-1], f"{values[-1]:.3f}", va="center", ha="left", fontsize=9, color="#374151")
    ax_top.set_xticks(x)
    ax_top.set_xticklabels(xlabels)
    ax_top.set_ylim(0.20, 0.75)
    ax_top.set_ylabel("Mean normalized physics error")
    ax_top.set_title("A. Pooled scaling behavior across 2, 4, and 8 qubits", loc="left", fontsize=12.5, fontweight="bold", pad=8)
    ax_top.legend(frameon=False, fontsize=9.2, loc="upper left")

    category_colors = {
        "circuit_evolution": "#b35d68",
        "measurement_prediction": "#d18a5a",
        "operator_composition": "#8295a8",
        "entanglement_classification": "#5f8d80",
    }
    category_labels = {
        "circuit_evolution": "Circuit evolution",
        "measurement_prediction": "Measurement prediction",
        "operator_composition": "Operator composition",
        "entanglement_classification": "Entanglement",
    }
    for category in ["circuit_evolution", "measurement_prediction", "operator_composition", "entanglement_classification"]:
        values = [s2["category"][category], s4["category"][category], s8["category"][category]]
        ax_bottom.plot(
            x,
            values,
            marker="o",
            markersize=7.3,
            linewidth=2.2,
            color=category_colors[category],
            label=category_labels[category],
        )
        ax_bottom.text(x[-1] + 0.18, values[-1], f"{values[-1]:.3f}", va="center", ha="left", fontsize=8.8, color="#374151")
    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels(xlabels)
    ax_bottom.set_ylim(0.0, 1.02)
    ax_bottom.set_ylabel("Mean normalized physics error")
    ax_bottom.set_title("B. Exact quantum dynamics degrade fastest as state size grows", loc="left", fontsize=12.5, fontweight="bold", pad=8)
    ax_bottom.legend(frameon=False, fontsize=9.0, loc="upper left", ncol=2)

    fig.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.09)
    out = ROOT / "artifacts" / "figures" / "combined_all_qubits_scaling.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
