#!/usr/bin/env python3

"""Compare two QuantumPhysEval runs with different qubit counts."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

from quantumphyseval import benchmark as qpe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare QuantumPhysEval runs with different qubit counts.")
    parser.add_argument("--baseline-results", required=True, help="Baseline results JSON, usually the 2-qubit run.")
    parser.add_argument("--variant-results", required=True, help="Variant results JSON, usually the larger-qubit run.")
    parser.add_argument("--summary-json", required=True, help="Output path for the comparison summary JSON.")
    parser.add_argument("--plot", required=True, help="Output path for the comparison plot.")
    parser.add_argument("--baseline-label", default="2 qubits")
    parser.add_argument("--variant-label", default="4 qubits")
    return parser.parse_args()


def load(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def usable(rows: Sequence[dict]) -> List[dict]:
    return [row for row in rows if row.get("d_phys") is not None]


def mean_error(rows: Sequence[dict], *, model: str = None, category: str = None) -> float:
    values = [
        float(row["d_phys"])
        for row in rows
        if (model is None or str(row["model"]) == model)
        and (category is None or str(row["category"]) == category)
    ]
    if not values:
        return float("nan")
    return float(np.mean(values))


def build_summary(baseline_rows: Sequence[dict], variant_rows: Sequence[dict]) -> Dict[str, object]:
    models = sorted(
        {str(row["model"]) for row in baseline_rows} | {str(row["model"]) for row in variant_rows},
        key=lambda name: (
            min(float(row["size"]) for row in list(baseline_rows) + list(variant_rows) if str(row["model"]) == name),
            name,
        ),
    )
    categories = list(qpe.CATEGORY_ORDER)
    varying_categories = [
        category for category in categories if category != "operator_composition"
    ]

    per_model = []
    per_model_category = []
    for model in models:
        base_mean = mean_error(baseline_rows, model=model)
        var_mean = mean_error(variant_rows, model=model)
        base_var_mean = mean_error(
            [row for row in baseline_rows if str(row["category"]) in varying_categories],
            model=model,
        )
        var_var_mean = mean_error(
            [row for row in variant_rows if str(row["category"]) in varying_categories],
            model=model,
        )
        per_model.append(
            {
                "model": model,
                "baseline_mean_d_phys": base_mean,
                "variant_mean_d_phys": var_mean,
                "delta_variant_minus_baseline": var_mean - base_mean,
                "baseline_mean_d_phys_varying_categories": base_var_mean,
                "variant_mean_d_phys_varying_categories": var_var_mean,
                "delta_variant_minus_baseline_varying_categories": var_var_mean - base_var_mean,
            }
        )
        for category in categories:
            base_cat = mean_error(baseline_rows, model=model, category=category)
            var_cat = mean_error(variant_rows, model=model, category=category)
            per_model_category.append(
                {
                    "model": model,
                    "category": category,
                    "baseline_mean_d_phys": base_cat,
                    "variant_mean_d_phys": var_cat,
                    "delta_variant_minus_baseline": var_cat - base_cat,
                }
            )

    overall = {
        "baseline_mean_d_phys": mean_error(baseline_rows),
        "variant_mean_d_phys": mean_error(variant_rows),
        "delta_variant_minus_baseline": mean_error(variant_rows) - mean_error(baseline_rows),
        "baseline_mean_d_phys_varying_categories": mean_error(
            [row for row in baseline_rows if str(row["category"]) in varying_categories]
        ),
        "variant_mean_d_phys_varying_categories": mean_error(
            [row for row in variant_rows if str(row["category"]) in varying_categories]
        ),
    }
    overall["delta_variant_minus_baseline_varying_categories"] = (
        overall["variant_mean_d_phys_varying_categories"] - overall["baseline_mean_d_phys_varying_categories"]
    )

    return {
        "models": models,
        "categories": categories,
        "varying_categories": varying_categories,
        "baseline_count": len(baseline_rows),
        "variant_count": len(variant_rows),
        "overall": overall,
        "per_model": per_model,
        "per_model_category": per_model_category,
    }


def save_plot(
    summary: Dict[str, object],
    plot_path: str,
    baseline_label: str,
    variant_label: str,
) -> None:
    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    models = list(summary["models"])
    per_model = {row["model"]: row for row in summary["per_model"]}
    per_model_category = summary["per_model_category"]

    fig = plt.figure(figsize=(11.2, 5.3), facecolor="white")
    grid = fig.add_gridspec(1, 2, width_ratios=[0.92, 1.08], wspace=0.28)
    ax_model = fig.add_subplot(grid[0, 0])
    ax_delta = fig.add_subplot(grid[0, 1])

    qpe.apply_plot_style(ax_model)
    qpe.apply_plot_style(ax_delta)

    y = np.arange(len(models), dtype=float)
    baseline = [per_model[model]["baseline_mean_d_phys"] for model in models]
    variant = [per_model[model]["variant_mean_d_phys"] for model in models]
    for idx in range(len(models)):
        ax_model.plot(
            [baseline[idx], variant[idx]],
            [idx, idx],
            color="#d0d7df",
            linewidth=2.4,
            solid_capstyle="round",
            zorder=1,
        )
    ax_model.scatter(
        baseline,
        y,
        s=72,
        color="#4f8fba",
        edgecolor="white",
        linewidth=0.9,
        label=baseline_label,
        zorder=2,
    )
    ax_model.scatter(
        variant,
        y,
        s=72,
        color="#c75c5c",
        edgecolor="white",
        linewidth=0.9,
        label=variant_label,
        zorder=3,
    )
    ax_model.set_yticks(y)
    ax_model.set_yticklabels([qpe.MODEL_LABELS.get(model, model) for model in models])
    ax_model.invert_yaxis()
    ax_model.set_xlabel("Mean normalized physics error")
    ax_model.set_title("A. Overall benchmark error", loc="left", fontsize=11.3, fontweight="bold", pad=7)
    ax_model.legend(frameon=False, fontsize=8.8, loc="lower right")

    categories = list(qpe.CATEGORY_ORDER)
    heat = np.zeros((len(models), len(categories)), dtype=float)
    for row_idx, model in enumerate(models):
        for col_idx, category in enumerate(categories):
            match = next(
                item for item in per_model_category
                if item["model"] == model and item["category"] == category
            )
            heat[row_idx, col_idx] = float(match["delta_variant_minus_baseline"])

    max_abs = max(0.05, float(np.max(np.abs(heat))))
    im = ax_delta.imshow(heat, cmap="RdBu_r", vmin=-max_abs, vmax=max_abs, aspect="auto")
    ax_delta.set_xticks(np.arange(len(categories)))
    ax_delta.set_xticklabels([qpe.SHORT_CATEGORY_LABELS[category] for category in categories])
    ax_delta.set_yticks(np.arange(len(models)))
    ax_delta.set_yticklabels([qpe.MODEL_LABELS.get(model, model) for model in models])
    ax_delta.set_title(
        f"B. {variant_label} minus {baseline_label} error",
        loc="left",
        fontsize=11.3,
        fontweight="bold",
        pad=7,
    )
    for spine in ax_delta.spines.values():
        spine.set_visible(False)
    ax_delta.tick_params(length=0)
    ax_delta.set_xticks(np.arange(-0.5, len(categories), 1), minor=True)
    ax_delta.set_yticks(np.arange(-0.5, len(models), 1), minor=True)
    ax_delta.grid(which="minor", color="white", linewidth=1.2)
    ax_delta.tick_params(which="minor", bottom=False, left=False)
    for row_idx in range(len(models)):
        for col_idx in range(len(categories)):
            value = heat[row_idx, col_idx]
            ax_delta.text(
                col_idx,
                row_idx,
                f"{value:+.2f}",
                ha="center",
                va="center",
                color="white" if abs(value) > 0.18 else "#16324f",
                fontsize=8.0,
                fontweight="bold",
            )
    cbar = fig.colorbar(im, ax=ax_delta, fraction=0.046, pad=0.04)
    cbar.set_label("Delta error", fontsize=9.5)
    cbar.ax.tick_params(labelsize=8.5)

    fig.subplots_adjust(left=0.14, right=0.98, top=0.89, bottom=0.16, wspace=0.28)
    fig.savefig(plot_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    baseline_rows = usable(load(args.baseline_results))
    variant_rows = usable(load(args.variant_results))
    summary = build_summary(baseline_rows, variant_rows)
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    save_plot(summary, args.plot, args.baseline_label, args.variant_label)
    print(f"Wrote {args.summary_json}")
    print(f"Wrote {args.plot}")


if __name__ == "__main__":
    main()
