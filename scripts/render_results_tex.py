#!/usr/bin/env python3

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render current experiment statistics into a LaTeX report fragment."
    )
    parser.add_argument(
        "--results-json",
        action="append",
        default=None,
        help="Path to a benchmark results JSON. Repeat to pool multiple runs.",
    )
    parser.add_argument(
        "--fit-json",
        default="artifacts/results/gpt4_family_fit.json",
        help="Path to the scaling-fit JSON.",
    )
    parser.add_argument(
        "--output-tex",
        default="paper/live_results_full.tex",
        help="Path to the generated LaTeX file.",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "paper"],
        default="full",
        help="Render the exhaustive live snapshot or a curated paper appendix fragment.",
    )
    return parser.parse_args()


def latex_escape(text: str) -> str:
    escaped = (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )
    return escaped


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values)


def sd(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((x - mu) ** 2 for x in values) / (len(values) - 1))


def quantile(values: List[float], p: float) -> float:
    if len(values) == 1:
        return values[0]
    idx = p * (len(values) - 1)
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return values[lo]
    weight = idx - lo
    return values[lo] * (1 - weight) + values[hi] * weight


def stats(values: Iterable[float]) -> Dict[str, float]:
    vals = sorted(float(x) for x in values)
    n = len(vals)
    mu = mean(vals)
    se = sd(vals) / math.sqrt(n) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mu,
        "median": quantile(vals, 0.5),
        "p90": quantile(vals, 0.9),
        "p95": quantile(vals, 0.95),
        "sd": sd(vals),
        "ci95_lo": mu - 1.96 * se,
        "ci95_hi": mu + 1.96 * se,
        "zero_rate": sum(1 for v in vals if v == 0.0) / n,
        "ge_0_5_rate": sum(1 for v in vals if v >= 0.5) / n,
        "full_error_rate": sum(1 for v in vals if v >= 0.999999) / n,
    }


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def pretty_model(model: str) -> str:
    return {
        "gpt-4o-mini": "GPT-4o mini",
        "gpt-4.1-mini": "GPT-4.1 mini",
        "gpt-4.1": "GPT-4.1",
        "gpt-5.1": "GPT-5.1",
        "gpt-5.2": "GPT-5.2",
        "gpt-5.4-mini": "GPT-5.4 mini",
        "gpt-5.4": "GPT-5.4",
        "claude-sonnet-4-20250514": "Claude Sonnet 4",
        "claude-opus-4-1-20250805": "Claude Opus 4.1",
    }.get(model, model)


def pretty_category(category: str) -> str:
    return {
        "circuit_evolution": "Circuit",
        "operator_composition": "Operator",
        "measurement_prediction": "Measurement",
        "entanglement_classification": "Entanglement",
    }.get(category, category)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def tabular(headers: List[str], rows: List[List[str]], spec: str) -> str:
    lines = [f"\\begin{{tabular}}{{{spec}}}", "\\hline"]
    lines.append(" & ".join(headers) + " \\\\")
    lines.append("\\hline")
    for row in rows:
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def booktabs_tabular(headers: List[str], rows: List[List[str]], spec: str) -> str:
    lines = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule"]
    lines.append(" & ".join(headers) + " \\\\")
    lines.append("\\midrule")
    for row in rows:
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def chunked(items: List[List[str]], chunk_size: int) -> List[List[List[str]]]:
    return [items[idx : idx + chunk_size] for idx in range(0, len(items), chunk_size)]


def add_chunked_table(
    lines: List[str],
    title: str,
    headers: List[str],
    rows: List[List[str]],
    spec: str,
    chunk_size: int,
) -> None:
    row_chunks = chunked(rows, chunk_size)
    if not row_chunks:
        lines.append(f"\\subsubsection*{{{title}}}")
        lines.append("No rows available.")
        return
    for idx, row_chunk in enumerate(row_chunks):
        lines.append("")
        heading = title if idx == 0 else f"{title} (cont.)"
        lines.append(f"\\subsubsection*{{{heading}}}")
        lines.append(tabular(headers=headers, rows=row_chunk, spec=spec))


def render_full_fragment(
    *,
    result_paths: List[str],
    fit_json_name: str,
    generated_at: str,
    results: List[dict],
    overall: Dict[str, float],
    model_stats: Dict[str, Dict[str, float]],
    depth_stats: Dict[int, Dict[str, float]],
    category_stats: Dict[str, Dict[str, float]],
    model_depth_stats: Dict[Tuple[str, int], Dict[str, float]],
    model_category_stats: Dict[Tuple[str, str], Dict[str, float]],
    model_category_depth_stats: Dict[Tuple[str, str, int], Dict[str, float]],
    best_model: str,
    worst_model: str,
    best_depth: int,
    hardest_category: str,
    easiest_category: str,
    alpha: float,
    beta: float,
    alpha_ci,
    beta_ci,
    pairwise_rows: List[List[str]],
    ranked_cells,
) -> str:
    lines: List[str] = []
    lines.append("% Auto-generated by update_results_tex.py. Do not edit manually.")
    lines.append(f"% Generated at {generated_at}")
    lines.append("\\subsection*{Running Results Snapshot}")
    source_text = ", ".join(
        f"\\texttt{{{latex_escape(Path(path).name)}}}" for path in result_paths
    )
    lines.append(
        "This fragment is generated directly from "
        f"{source_text} and "
        f"\\texttt{{{latex_escape(fit_json_name)}}}."
    )
    lines.append("")
    lines.append(
        f"The current finished run contains {len(results)} usable evaluations. "
        f"The overall mean normalized error is {fmt(overall['mean'])}, with median "
        f"{fmt(overall['median'])}, 90th percentile {fmt(overall['p90'])}, and 95th "
        f"percentile {fmt(overall['p95'])}. Exact-correct outputs occur on "
        f"{fmt(overall['zero_rate'])} of samples, while {fmt(overall['full_error_rate'])} "
        "of samples are effectively full-error failures."
    )
    lines.append(
        f"The best model by mean error is \\texttt{{{latex_escape(best_model)}}} "
        f"({fmt(model_stats[best_model]['mean'])}), while the weakest is "
        f"\\texttt{{{latex_escape(worst_model)}}} ({fmt(model_stats[worst_model]['mean'])}). "
        f"The best depth setting is {best_depth} with mean error "
        f"{fmt(depth_stats[best_depth]['mean'])}. The hardest category is "
        f"\\texttt{{{latex_escape(hardest_category)}}} ({fmt(category_stats[hardest_category]['mean'])}), "
        f"and the easiest is \\texttt{{{latex_escape(easiest_category)}}} "
        f"({fmt(category_stats[easiest_category]['mean'])})."
    )
    if alpha is not None and beta is not None:
        alpha_text = f"{fmt(alpha)}"
        beta_text = f"{fmt(beta)}"
        if alpha_ci and beta_ci:
            alpha_text += f" [{fmt(alpha_ci[0])}, {fmt(alpha_ci[1])}]"
            beta_text += f" [{fmt(beta_ci[0])}, {fmt(beta_ci[1])}]"
        lines.append(
            f"The fitted scaling law is $d_{{phys}} \\approx k S^{{-{fmt(alpha)}}} D^{{{fmt(beta)}}}$, "
            f"with $\\alpha={alpha_text}$ and $\\beta={beta_text}$."
        )

    lines.append("")
    lines.append("\\subsubsection*{Overall Summary}")
    lines.append(
        tabular(
            headers=["Metric", "Value"],
            rows=[
                ["Usable evaluations", str(len(results))],
                ["Mean $d_{phys}$", fmt(overall["mean"])],
                ["Median $d_{phys}$", fmt(overall["median"])],
                ["$p_{90}$", fmt(overall["p90"])],
                ["$p_{95}$", fmt(overall["p95"])],
                ["Exact-correct rate", fmt(overall["zero_rate"])],
                ["$d_{phys} \\ge 0.5$ rate", fmt(overall["ge_0_5_rate"])],
                ["Full-error rate", fmt(overall["full_error_rate"])],
            ],
            spec="lr",
        )
    )

    lines.append("")
    lines.append("\\subsubsection*{By Model}")
    model_rows: List[List[str]] = []
    for model in sorted(model_stats):
        entry = model_stats[model]
        model_rows.append(
            [
                latex_escape(model),
                str(entry["n"]),
                fmt(entry["mean"]),
                fmt(entry["median"]),
                f"[{fmt(entry['ci95_lo'])}, {fmt(entry['ci95_hi'])}]",
                fmt(entry["zero_rate"]),
                fmt(entry["full_error_rate"]),
            ]
        )
    lines.append(
        tabular(
            headers=["Model", "$n$", "Mean", "Median", "95\\% CI", "Zero rate", "Full-error rate"],
            rows=model_rows,
            spec="lrrrrrr",
        )
    )

    lines.append("")
    lines.append("\\subsubsection*{By Depth}")
    depth_rows: List[List[str]] = []
    for depth in sorted(depth_stats):
        entry = depth_stats[depth]
        depth_rows.append(
            [
                str(depth),
                str(entry["n"]),
                fmt(entry["mean"]),
                fmt(entry["median"]),
                f"[{fmt(entry['ci95_lo'])}, {fmt(entry['ci95_hi'])}]",
                fmt(entry["zero_rate"]),
                fmt(entry["full_error_rate"]),
            ]
        )
    lines.append(
        tabular(
            headers=["Depth", "$n$", "Mean", "Median", "95\\% CI", "Zero rate", "Full-error rate"],
            rows=depth_rows,
            spec="rrrrrrr",
        )
    )

    lines.append("")
    lines.append("\\subsubsection*{By Category}")
    category_rows: List[List[str]] = []
    for category in sorted(category_stats):
        entry = category_stats[category]
        category_rows.append(
            [
                latex_escape(category),
                str(entry["n"]),
                fmt(entry["mean"]),
                fmt(entry["median"]),
                fmt(entry["p90"]),
                fmt(entry["zero_rate"]),
                fmt(entry["full_error_rate"]),
            ]
        )
    lines.append(
        tabular(
            headers=["Category", "$n$", "Mean", "Median", "$p_{90}$", "Zero rate", "Full-error rate"],
            rows=category_rows,
            spec="lrrrrrr",
        )
    )

    if pairwise_rows:
        lines.append("")
        lines.append("\\subsubsection*{Pairwise Model Improvements}")
        lines.append(
            tabular(
                headers=["Comparison", "Absolute gain", "Relative gain"],
                rows=pairwise_rows,
                spec="lrr",
            )
        )

    lines.append("")
    lines.append("\\subsubsection*{Best and Worst Model-Category Cells}")
    lines.append("\\begin{itemize}")
    for value, model, category, lo, hi in ranked_cells[:3]:
        lines.append(
            f"\\item Best: \\texttt{{{latex_escape(model)}}} on "
            f"\\texttt{{{latex_escape(category)}}} with mean {fmt(value)} "
            f"and 95\\% CI [{fmt(lo)}, {fmt(hi)}]."
        )
    for value, model, category, lo, hi in ranked_cells[-3:]:
        lines.append(
            f"\\item Worst: \\texttt{{{latex_escape(model)}}} on "
            f"\\texttt{{{latex_escape(category)}}} with mean {fmt(value)} "
            f"and 95\\% CI [{fmt(lo)}, {fmt(hi)}]."
        )
    lines.append("\\end{itemize}")

    model_depth_rows: List[List[str]] = []
    for (model, depth) in sorted(model_depth_stats):
        entry = model_depth_stats[(model, depth)]
        model_depth_rows.append(
            [
                latex_escape(model),
                str(depth),
                fmt(entry["mean"]),
                fmt(entry["median"]),
                fmt(entry["zero_rate"]),
                fmt(entry["full_error_rate"]),
            ]
        )
    add_chunked_table(
        lines=lines,
        title="Model by Depth",
        headers=["Model", "Depth", "Mean", "Median", "Zero rate", "Full-error rate"],
        rows=model_depth_rows,
        spec="lrrrrr",
        chunk_size=12,
    )

    model_category_rows: List[List[str]] = []
    for (model, category) in sorted(model_category_stats):
        entry = model_category_stats[(model, category)]
        model_category_rows.append(
            [
                latex_escape(model),
                latex_escape(category),
                fmt(entry["mean"]),
                fmt(entry["median"]),
                f"[{fmt(entry['ci95_lo'])}, {fmt(entry['ci95_hi'])}]",
                fmt(entry["zero_rate"]),
                fmt(entry["full_error_rate"]),
            ]
        )
    add_chunked_table(
        lines=lines,
        title="Model by Category",
        headers=["Model", "Category", "Mean", "Median", "95\\% CI", "Zero rate", "Full-error rate"],
        rows=model_category_rows,
        spec="llrrrrr",
        chunk_size=14,
    )

    detailed_rows: List[List[str]] = []
    for (model, category, depth) in sorted(model_category_depth_stats):
        entry = model_category_depth_stats[(model, category, depth)]
        detailed_rows.append(
            [
                latex_escape(model),
                latex_escape(category),
                str(depth),
                str(entry["n"]),
                fmt(entry["mean"]),
                fmt(entry["zero_rate"]),
                fmt(entry["ge_0_5_rate"]),
                fmt(entry["full_error_rate"]),
            ]
        )
    add_chunked_table(
        lines=lines,
        title="Detailed Cell Rates (Model, Category, Depth)",
        headers=[
            "Model",
            "Category",
            "Depth",
            "$n$",
            "Mean",
            "Zero rate",
            "$d_{phys} \\ge 0.5$",
            "Full-error rate",
        ],
        rows=detailed_rows,
        spec="llrrrrrr",
        chunk_size=18,
    )
    return "\n".join(lines) + "\n"


def render_paper_fragment(
    *,
    result_paths: List[str],
    fit_json_name: str,
    results: List[dict],
    overall: Dict[str, float],
    model_stats: Dict[str, Dict[str, float]],
    depth_stats: Dict[int, Dict[str, float]],
    category_stats: Dict[str, Dict[str, float]],
    model_category_stats: Dict[Tuple[str, str], Dict[str, float]],
    best_model: str,
    worst_model: str,
    alpha: float,
    beta: float,
    alpha_ci,
    beta_ci,
) -> str:
    lines: List[str] = []
    lines.append("% Auto-generated paper appendix fragment.")
    lines.append("\\subsection*{Appendix Snapshot}")
    lines.append(
        f"This appendix summarizes the pooled OpenAI GPT and Anthropic Claude benchmark over {len(results)} scored responses. "
        f"The full machine-generated statistics remain available separately in "
        f"\\texttt{{appendix\\_snapshot\\_full.tex}}."
    )
    if alpha is not None and beta is not None and alpha_ci and beta_ci:
        lines.append(
            f"The pooled fit yields $\\alpha={fmt(alpha)}$ with 95\\% interval "
            f"$[{fmt(alpha_ci[0])}, {fmt(alpha_ci[1])}]$ and $\\beta={fmt(beta)}$ with interval "
            f"$[{fmt(beta_ci[0])}, {fmt(beta_ci[1])}]$."
        )
    lines.append("")
    lines.append("\\subsubsection*{Top-Line Summary}")
    lines.append("\\begin{center}")
    lines.append(booktabs_tabular(
        headers=["Metric", "Value"],
        rows=[
            ["Usable evaluations", str(len(results))],
            ["Mean $d_{phys}$", fmt(overall["mean"])],
            ["Median $d_{phys}$", fmt(overall["median"])],
            ["Exact-correct rate", fmt(overall["zero_rate"])],
            ["$d_{phys} \\ge 0.5$ rate", fmt(overall["ge_0_5_rate"])],
            ["Full-error rate", fmt(overall["full_error_rate"])],
            ["Best model", pretty_model(best_model)],
            ["Weakest model", pretty_model(worst_model)],
        ],
        spec="lr",
    ))
    lines.append("\\end{center}")

    model_rows: List[List[str]] = []
    for model in sorted(model_stats, key=lambda name: model_stats[name]["mean"]):
        entry = model_stats[model]
        model_rows.append(
            [
                pretty_model(model),
                fmt(entry["mean"]),
                f"[{fmt(entry['ci95_lo'])}, {fmt(entry['ci95_hi'])}]",
                fmt(entry["zero_rate"]),
                fmt(entry["full_error_rate"]),
            ]
        )
    lines.append("")
    lines.append("\\subsubsection*{Model Ranking}")
    lines.append("\\begin{center}")
    lines.append(booktabs_tabular(
        headers=["Model", "Mean", "95\\% CI", "Exact", "Full"],
        rows=model_rows,
        spec="lrrrr",
    ))
    lines.append("\\end{center}")

    lines.append("")
    lines.append("\\subsubsection*{Depth and Category Breakdown}")
    depth_rows = [
        [
            str(depth),
            fmt(depth_stats[depth]["mean"]),
            fmt(depth_stats[depth]["zero_rate"]),
            fmt(depth_stats[depth]["full_error_rate"]),
        ]
        for depth in sorted(depth_stats)
    ]
    category_rows = [
        [
            pretty_category(category),
            fmt(category_stats[category]["mean"]),
            fmt(category_stats[category]["zero_rate"]),
            fmt(category_stats[category]["full_error_rate"]),
        ]
        for category in sorted(category_stats, key=lambda name: category_stats[name]["mean"], reverse=True)
    ]
    lines.append("\\noindent\\begin{minipage}[t]{0.42\\linewidth}")
    lines.append("\\centering")
    lines.append("\\textbf{By depth}\\\\[0.25em]")
    lines.append(booktabs_tabular(
        headers=["Depth", "Mean", "Exact", "Full"],
        rows=depth_rows,
        spec="rrrr",
    ))
    lines.append("\\end{minipage}\\hfill")
    lines.append("\\begin{minipage}[t]{0.54\\linewidth}")
    lines.append("\\centering")
    lines.append("\\textbf{By category}\\\\[0.25em]")
    lines.append(booktabs_tabular(
        headers=["Category", "Mean", "Exact", "Full"],
        rows=category_rows,
        spec="lrrr",
    ))
    lines.append("\\end{minipage}")

    model_category_rows: List[List[str]] = []
    categories = [
        "circuit_evolution",
        "measurement_prediction",
        "operator_composition",
        "entanglement_classification",
    ]
    for model in sorted(model_stats, key=lambda name: model_stats[name]["mean"]):
        model_category_rows.append(
            [pretty_model(model)]
            + [fmt(model_category_stats[(model, category)]["mean"]) for category in categories]
        )
    lines.append("")
    lines.append("\\subsubsection*{Model-by-Category Means}")
    lines.append("\\begin{center}")
    lines.append(booktabs_tabular(
        headers=["Model", "Circuit", "Measurement", "Operator", "Entanglement"],
        rows=model_category_rows,
        spec="lrrrr",
    ))
    lines.append("\\end{center}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    result_paths = args.results_json or ["artifacts/results/gpt4_family_results.json"]
    results = []
    for path in result_paths:
        results.extend(row for row in load_json(path) if row.get("d_phys") is not None)
    fit = load_json(args.fit_json)

    by_model: Dict[str, List[float]] = defaultdict(list)
    by_depth: Dict[int, List[float]] = defaultdict(list)
    by_category: Dict[str, List[float]] = defaultdict(list)
    by_model_depth: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    by_model_category: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    by_model_category_depth: Dict[Tuple[str, str, int], List[float]] = defaultdict(list)

    for row in results:
        value = float(row["d_phys"])
        by_model[str(row["model"])].append(value)
        by_depth[int(row["depth"])].append(value)
        by_category[str(row["category"])].append(value)
        by_model_depth[(str(row["model"]), int(row["depth"]))].append(value)
        by_model_category[(str(row["model"]), str(row["category"]))].append(value)
        by_model_category_depth[(str(row["model"]), str(row["category"]), int(row["depth"]))].append(value)

    overall = stats(row["d_phys"] for row in results)
    model_stats = {key: stats(vals) for key, vals in by_model.items()}
    depth_stats = {key: stats(vals) for key, vals in by_depth.items()}
    category_stats = {key: stats(vals) for key, vals in by_category.items()}
    model_depth_stats = {key: stats(vals) for key, vals in by_model_depth.items()}
    model_category_stats = {key: stats(vals) for key, vals in by_model_category.items()}
    model_category_depth_stats = {key: stats(vals) for key, vals in by_model_category_depth.items()}

    model_order = sorted(model_stats, key=lambda key: model_stats[key]["mean"])
    best_model = model_order[0]
    worst_model = model_order[-1]
    best_depth = min(depth_stats, key=lambda key: depth_stats[key]["mean"])
    hardest_category = max(category_stats, key=lambda key: category_stats[key]["mean"])
    easiest_category = min(category_stats, key=lambda key: category_stats[key]["mean"])

    pairwise_rows = []
    ranked_worst_to_best = sorted(model_stats, key=lambda key: model_stats[key]["mean"], reverse=True)
    for worse, better in zip(ranked_worst_to_best, ranked_worst_to_best[1:]):
        base_mean = model_stats[worse]["mean"]
        abs_improvement = base_mean - model_stats[better]["mean"]
        rel_improvement = abs_improvement / base_mean if base_mean else 0.0
        pairwise_rows.append(
            [
                f"{latex_escape(better)} vs {latex_escape(worse)}",
                fmt(abs_improvement),
                fmt(rel_improvement),
            ]
        )

    ranked_cells = sorted(
        (
            (entry["mean"], model, category, entry["ci95_lo"], entry["ci95_hi"])
            for (model, category), entry in model_category_stats.items()
        ),
        key=lambda item: item[0],
    )

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    alpha = fit.get("alpha")
    beta = fit.get("beta")
    alpha_ci = fit.get("bootstrap", {}).get("alpha_ci95") if fit.get("bootstrap") else None
    beta_ci = fit.get("bootstrap", {}).get("beta_ci95") if fit.get("bootstrap") else None
    if args.mode == "paper":
        rendered = render_paper_fragment(
            result_paths=result_paths,
            fit_json_name=Path(args.fit_json).name,
            results=results,
            overall=overall,
            model_stats=model_stats,
            depth_stats=depth_stats,
            category_stats=category_stats,
            model_category_stats=model_category_stats,
            best_model=best_model,
            worst_model=worst_model,
            alpha=alpha,
            beta=beta,
            alpha_ci=alpha_ci,
            beta_ci=beta_ci,
        )
    else:
        rendered = render_full_fragment(
            result_paths=result_paths,
            fit_json_name=Path(args.fit_json).name,
            generated_at=generated_at,
            results=results,
            overall=overall,
            model_stats=model_stats,
            depth_stats=depth_stats,
            category_stats=category_stats,
            model_depth_stats=model_depth_stats,
            model_category_stats=model_category_stats,
            model_category_depth_stats=model_category_depth_stats,
            best_model=best_model,
            worst_model=worst_model,
            best_depth=best_depth,
            hardest_category=hardest_category,
            easiest_category=easiest_category,
            alpha=alpha,
            beta=beta,
            alpha_ci=alpha_ci,
            beta_ci=beta_ci,
            pairwise_rows=pairwise_rows,
            ranked_cells=ranked_cells,
        )

    output_path = Path(args.output_tex)
    output_path.write_text(rendered, encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
