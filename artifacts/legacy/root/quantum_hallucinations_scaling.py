#!/usr/bin/env python3

import argparse
import ast
import json
import os
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI
from tqdm import tqdm


DEFAULT_MODEL_SIZES: Dict[str, float] = {
    "gpt-4": 1e10,
    "gpt-4o-mini": 2e9,
}
DEFAULT_REASONING_DEPTHS = (1, 2, 4)
EPS = 1e-8
MODEL_COLORS = {
    "gpt-4o-mini": "#d95f02",
    "gpt-4.1-mini": "#7570b3",
    "gpt-4.1": "#1b9e77",
}
CATEGORY_COLORS = {
    "state_construction": "#264653",
    "operator_synthesis": "#e76f51",
    "circuit_evolution": "#2a9d8f",
    "entanglement_reasoning": "#e9c46a",
}
CATEGORY_LABELS = {
    "state_construction": "State",
    "operator_synthesis": "Operator",
    "circuit_evolution": "Circuit",
    "entanglement_reasoning": "Entanglement",
}


@dataclass
class PromptSpec:
    category: str
    prompt: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate a simple scaling law for physics-violation rates in quantum prompts."
    )
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument(
        "--reasoning-depths",
        type=int,
        nargs="+",
        default=list(DEFAULT_REASONING_DEPTHS),
    )
    parser.add_argument(
        "--model",
        action="append",
        metavar="NAME:SIZE",
        help="Override models as repeated NAME:SIZE pairs, e.g. --model gpt-4:1e10",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--results-json",
        default="quantum_hallucinations_neurips.json",
    )
    parser.add_argument(
        "--fit-json",
        default="quantum_hallucinations_fit.json",
    )
    parser.add_argument(
        "--plot-path",
        default="scaling_law_neurips.png",
    )
    parser.add_argument(
        "--dashboard-path",
        default="quantum_hallucinations_dashboard.png",
    )
    parser.add_argument(
        "--reuse-results-json",
        help="Skip API querying and reuse an existing results JSON file.",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Display the plot interactively after saving it.",
    )
    return parser.parse_args()


def parse_models(model_args: Optional[Sequence[str]]) -> Dict[str, float]:
    if not model_args:
        return dict(DEFAULT_MODEL_SIZES)

    parsed: Dict[str, float] = {}
    for item in model_args:
        if ":" not in item:
            raise ValueError(f"Invalid --model entry: {item!r}")
        name, size = item.split(":", 1)
        parsed[name] = float(size)
    return parsed


def generate_prompts(n_samples: int, seed: int) -> List[PromptSpec]:
    categories = [
        "state_construction",
        "operator_synthesis",
        "circuit_evolution",
        "entanglement_reasoning",
    ]
    rng = np.random.default_rng(seed)
    prompts: List[PromptSpec] = []
    sampled_categories = [
        categories[idx % len(categories)] for idx in range(n_samples)
    ]
    rng.shuffle(sampled_categories)

    for cat in sampled_categories:
        if cat == "state_construction":
            prompt = (
                "Construct a valid 2-qubit quantum state as a Python list of 4 complex "
                "numbers. Use decimal numeric literals only, with no symbolic expressions. "
                "Return only the list."
            )
        elif cat == "operator_synthesis":
            prompt = (
                "Write a 2x2 unitary matrix as a Python 2x2 list of complex numbers. "
                "Use decimal numeric literals only, with no symbolic expressions. "
                "Return only the matrix."
            )
        elif cat == "circuit_evolution":
            prompt = (
                "Apply Hadamard to qubit 1 and CNOT to qubits 1,2 on |00>. "
                "Return the final state as a Python list of 4 complex numbers only, "
                "using decimal numeric literals."
            )
        else:
            prompt = (
                "Check whether the state (|00> + |11>)/sqrt(2) is entangled. "
                "Return only True or False."
            )
        prompts.append(PromptSpec(category=cat, prompt=prompt))

    return prompts


def require_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return api_key


def query_openai(
    client: OpenAI,
    prompt: str,
    model: str,
    temperature: float,
) -> str:
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {
                "role": "system",
                "content": (
                    "Return concise machine-readable answers only. "
                    "Do not add explanations or markdown."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content
    return (content or "").strip()


def normalize_complex_literal(text: str) -> str:
    return (
        text.replace("i", "j")
        .replace("I", "j")
        .replace("−", "-")
        .replace("\n", " ")
    )


def safe_eval_numeric(expr: str):
    node = ast.parse(expr, mode="eval")

    def visit(current):
        if isinstance(current, ast.Expression):
            return visit(current.body)
        if isinstance(current, ast.List):
            return [visit(item) for item in current.elts]
        if isinstance(current, ast.Tuple):
            return tuple(visit(item) for item in current.elts)
        if isinstance(current, ast.Constant):
            if isinstance(current.value, (int, float, complex, bool)):
                return current.value
            raise ValueError(f"Unsupported constant: {current.value!r}")
        if isinstance(current, ast.UnaryOp) and isinstance(current.op, (ast.UAdd, ast.USub)):
            operand = visit(current.operand)
            return +operand if isinstance(current.op, ast.UAdd) else -operand
        if isinstance(current, ast.BinOp) and isinstance(
            current.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
        ):
            left = visit(current.left)
            right = visit(current.right)
            if isinstance(current.op, ast.Add):
                return left + right
            if isinstance(current.op, ast.Sub):
                return left - right
            if isinstance(current.op, ast.Mult):
                return left * right
            if isinstance(current.op, ast.Div):
                return left / right
            return left ** right
        raise ValueError(f"Unsupported expression: {ast.dump(current)}")

    return visit(node)


def parse_python_literal(text: str):
    cleaned = normalize_complex_literal(text)
    return safe_eval_numeric(cleaned)


def parse_complex_list(text: str) -> np.ndarray:
    try:
        value = parse_python_literal(text)
        arr = np.asarray(value, dtype=complex).reshape(-1)
        return arr
    except Exception:
        return np.array([], dtype=complex)


def parse_matrix(text: str) -> np.ndarray:
    try:
        value = parse_python_literal(text)
        arr = np.asarray(value, dtype=complex)
        if arr.ndim != 2:
            return np.empty((0, 0), dtype=complex)
        return arr
    except Exception:
        return np.empty((0, 0), dtype=complex)


def violation_metric(category: str, output: str) -> float:
    if category in {"state_construction", "circuit_evolution"}:
        state = parse_complex_list(output)
        if state.shape != (4,):
            return 1.0
        norm = float(np.sum(np.abs(state) ** 2))
        return abs(norm - 1.0)

    if category == "operator_synthesis":
        mat = parse_matrix(output)
        if mat.shape != (2, 2):
            return 2.0
        identity = np.eye(2, dtype=complex)
        return float(np.linalg.norm(mat.conj().T @ mat - identity))

    entangled_reported = output.strip().lower() == "true"
    correct = True
    return 0.0 if entangled_reported == correct else 1.0


def run_scaling_pipeline(
    client: OpenAI,
    models: Dict[str, float],
    depths: Sequence[int],
    n_samples: int,
    seed: int,
    temperature: float,
    results_json: str,
) -> List[dict]:
    prompts = generate_prompts(n_samples=n_samples, seed=seed)
    results: List[dict] = []

    for model_name, depth in tqdm(list(product(models.keys(), depths)), desc="Model/depth"):
        size = models[model_name]
        for prompt_spec in prompts:
            prompt_full = (
                f"{prompt_spec.prompt} Use exactly {depth} reasoning steps internally, "
                "but return only the final answer."
            )
            try:
                output_text = query_openai(
                    client=client,
                    prompt=prompt_full,
                    model=model_name,
                    temperature=temperature,
                )
                api_error = None
            except Exception as exc:
                output_text = ""
                api_error = str(exc)

            d_phys = violation_metric(prompt_spec.category, output_text) if not api_error else np.nan
            results.append(
                {
                    "model": model_name,
                    "size": size,
                    "depth": depth,
                    "category": prompt_spec.category,
                    "prompt": prompt_spec.prompt,
                    "output": output_text,
                    "d_phys": None if np.isnan(d_phys) else float(d_phys),
                    "api_error": api_error,
                }
            )

    with open(results_json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    return results


def fit_scaling_law(results: Sequence[dict]) -> Tuple[float, float, float]:
    usable = [r for r in results if r.get("d_phys") is not None]
    if len(usable) < 3:
        raise RuntimeError("Need at least 3 successful samples to fit the scaling law.")

    sizes = np.array([r["size"] for r in usable], dtype=float)
    depths = np.array([r["depth"] for r in usable], dtype=float)
    violations = np.array([max(r["d_phys"], EPS) for r in usable], dtype=float)
    design = np.column_stack(
        [
            np.ones_like(sizes),
            -np.log(sizes),
            np.log(depths),
        ]
    )
    target = np.log(violations)
    coeffs, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    log_k, alpha, beta = coeffs
    k = float(np.exp(log_k))
    return k, float(alpha), float(beta)


def save_fit_and_plot(
    results: Sequence[dict],
    fit_params: Tuple[float, float, float],
    fit_json: str,
    plot_path: str,
    show_plot: bool,
) -> None:
    usable = [r for r in results if r.get("d_phys") is not None]
    sizes = np.array([r["size"] for r in usable], dtype=float)
    depths = np.array([r["depth"] for r in usable], dtype=float)
    violations = np.array([max(r["d_phys"], EPS) for r in usable], dtype=float)
    k, alpha, beta = fit_params

    def scaling_func(size_vals, depth_val):
        return k * (size_vals ** (-alpha)) * (depth_val ** beta)

    with open(fit_json, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "k": k,
                "alpha": alpha,
                "beta": beta,
                "equation": "d_phys = k * S^(-alpha) * D^(beta)",
                "fit_method": "log_linear_least_squares",
            },
            handle,
            indent=2,
        )

    plt.figure(figsize=(8, 5))
    for depth in np.unique(depths):
        mask = depths == depth
        plt.scatter(sizes[mask], violations[mask], label=f"Depth={int(depth)}", alpha=0.8)
        span = np.geomspace(np.min(sizes[mask]), np.max(sizes[mask]), 100)
        plt.plot(span, scaling_func(span, depth), "--")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Model size S")
    plt.ylabel("Violation metric d_phys")
    plt.title("Quantum Hallucinations Scaling Law")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    if show_plot:
        plt.show()
    plt.close()


def save_dashboard_plot(
    results: Sequence[dict],
    fit_params: Tuple[float, float, float],
    dashboard_path: str,
) -> None:
    usable = [r for r in results if r.get("d_phys") is not None]
    if not usable:
        raise RuntimeError("No usable results available for dashboard plot.")

    k, alpha, beta = fit_params
    models = sorted({r["model"] for r in usable}, key=lambda name: min(r["size"] for r in usable if r["model"] == name))
    categories = [
        "state_construction",
        "operator_synthesis",
        "circuit_evolution",
        "entanglement_reasoning",
    ]
    depths = sorted({int(r["depth"]) for r in usable})

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax_scatter, ax_bars, ax_strip, ax_heat = axes.flatten()

    sizes = np.array([r["size"] for r in usable], dtype=float)
    violation_vals = np.array([max(r["d_phys"], EPS) for r in usable], dtype=float)
    depth_vals = np.array([r["depth"] for r in usable], dtype=float)

    def scaling_func(size_vals, depth_val):
        return k * (size_vals ** (-alpha)) * (depth_val ** beta)

    for depth in depths:
        mask = depth_vals == depth
        ax_scatter.scatter(
            sizes[mask],
            violation_vals[mask],
            s=40,
            alpha=0.85,
            label=f"Depth {depth}",
        )
        span = np.geomspace(np.min(sizes[mask]), np.max(sizes[mask]), 100)
        ax_scatter.plot(span, scaling_func(span, depth), "--", linewidth=1.6)
    ax_scatter.set_xscale("log")
    ax_scatter.set_yscale("log")
    ax_scatter.set_title("Scaling Scatter")
    ax_scatter.set_xlabel("Model size S")
    ax_scatter.set_ylabel("Violation d_phys")
    ax_scatter.legend(frameon=False)

    x = np.arange(len(categories))
    width = 0.24
    for idx, model in enumerate(models):
        means = []
        for category in categories:
            subset = [
                max(r["d_phys"], EPS)
                for r in usable
                if r["model"] == model and r["category"] == category
            ]
            means.append(float(np.mean(subset)) if subset else EPS)
        offset = (idx - (len(models) - 1) / 2.0) * width
        ax_bars.bar(
            x + offset,
            means,
            width=width,
            color=MODEL_COLORS.get(model, None),
            label=model,
        )
    ax_bars.set_yscale("log")
    ax_bars.set_xticks(x)
    ax_bars.set_xticklabels([CATEGORY_LABELS[c] for c in categories])
    ax_bars.set_title("Mean Violation by Category")
    ax_bars.set_ylabel("Mean d_phys")
    ax_bars.legend(frameon=False)

    rng = np.random.default_rng(13)
    for model_index, model in enumerate(models):
        subset = [r for r in usable if r["model"] == model]
        jitter = rng.uniform(-0.18, 0.18, size=len(subset))
        for point_index, row in enumerate(subset):
            ax_strip.scatter(
                model_index + jitter[point_index],
                max(row["d_phys"], EPS),
                s=55,
                alpha=0.9,
                color=CATEGORY_COLORS[row["category"]],
                edgecolors="white",
                linewidths=0.5,
            )
    ax_strip.set_yscale("log")
    ax_strip.set_xticks(np.arange(len(models)))
    ax_strip.set_xticklabels(models)
    ax_strip.set_title("Per-sample Violations")
    ax_strip.set_ylabel("d_phys")
    category_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=CATEGORY_LABELS[category],
            markerfacecolor=CATEGORY_COLORS[category],
            markersize=8,
        )
        for category in categories
    ]
    ax_strip.legend(handles=category_handles, frameon=False, loc="upper right")

    heat = np.zeros((len(models), len(categories)), dtype=float)
    labels = []
    for row_idx, model in enumerate(models):
        row_labels = []
        for col_idx, category in enumerate(categories):
            subset = [
                max(r["d_phys"], EPS)
                for r in usable
                if r["model"] == model and r["category"] == category
            ]
            value = float(np.mean(subset)) if subset else EPS
            heat[row_idx, col_idx] = np.log10(value)
            row_labels.append(f"{value:.2e}")
        labels.append(row_labels)
    im = ax_heat.imshow(heat, cmap="magma", aspect="auto")
    ax_heat.set_xticks(np.arange(len(categories)))
    ax_heat.set_xticklabels([CATEGORY_LABELS[c] for c in categories], rotation=20, ha="right")
    ax_heat.set_yticks(np.arange(len(models)))
    ax_heat.set_yticklabels(models)
    ax_heat.set_title("Mean log10(d_phys) Heatmap")
    for row_idx in range(len(models)):
        for col_idx in range(len(categories)):
            ax_heat.text(
                col_idx,
                row_idx,
                labels[row_idx][col_idx],
                ha="center",
                va="center",
                color="white",
                fontsize=8,
            )
    fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04, label="log10 mean d_phys")

    fig.suptitle("Quantum Hallucinations Results Dashboard", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(dashboard_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.reuse_results_json:
        with open(args.reuse_results_json, "r", encoding="utf-8") as handle:
            results = json.load(handle)
    else:
        require_api_key()
        client = OpenAI()
        models = parse_models(args.model)
        results = run_scaling_pipeline(
            client=client,
            models=models,
            depths=args.reasoning_depths,
            n_samples=args.n_samples,
            seed=args.seed,
            temperature=args.temperature,
            results_json=args.results_json,
        )
    fit_params = fit_scaling_law(results)
    save_fit_and_plot(
        results=results,
        fit_params=fit_params,
        fit_json=args.fit_json,
        plot_path=args.plot_path,
        show_plot=args.show_plot,
    )
    save_dashboard_plot(
        results=results,
        fit_params=fit_params,
        dashboard_path=args.dashboard_path,
    )

    k, alpha, beta = fit_params
    print(f"Scaling law fit: d_phys = {k:.6g} * S^-{alpha:.6g} * D^{beta:.6g}")
    print(f"Saved raw results to {args.results_json}")
    print(f"Saved fit params to {args.fit_json}")
    print(f"Saved plot to {args.plot_path}")
    print(f"Saved dashboard to {args.dashboard_path}")


if __name__ == "__main__":
    main()
