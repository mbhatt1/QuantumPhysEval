#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quantumphyseval import benchmark as qhp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build pooled OpenAI GPT / Anthropic Claude benchmark artifacts from multiple result files."
    )
    parser.add_argument(
        "--results-json",
        action="append",
        default=None,
        help="Input results JSON. Repeat to pool multiple completed runs.",
    )
    parser.add_argument(
        "--combined-results-json",
        default="artifacts/results/combined_quantum_results.json",
        help="Path to the merged raw results JSON.",
    )
    parser.add_argument(
        "--summary-json",
        default="artifacts/results/combined_quantum_summary.json",
        help="Path to the pooled summary JSON.",
    )
    parser.add_argument(
        "--fit-json",
        default="artifacts/results/combined_quantum_fit.json",
        help="Path to the pooled scaling-fit JSON.",
    )
    parser.add_argument(
        "--main-plot",
        default="artifacts/figures/combined_quantum_scaling.png",
        help="Path to the pooled scaling figure.",
    )
    parser.add_argument(
        "--overview-plot",
        default="artifacts/figures/quantumphys_eval_overview.png",
        help="Path to the benchmark overview figure.",
    )
    parser.add_argument(
        "--takeaway-plot",
        default="artifacts/figures/combined_quantum_takeaways.png",
        help="Path to the direct takeaways figure.",
    )
    parser.add_argument(
        "--breakdown-plot",
        default="artifacts/figures/combined_quantum_breakdown.png",
        help="Path to the pooled breakdown figure.",
    )
    parser.add_argument(
        "--ablation-plot",
        default="artifacts/figures/combined_quantum_ablations.png",
        help="Path to the pooled ablation figure.",
    )
    parser.add_argument(
        "--output-tex",
        default="paper/appendix_snapshot_full.tex",
        help="Path to the pooled LaTeX statistics snapshot.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="Number of bootstrap resamples for the pooled fit.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for bootstrap resampling.",
    )
    return parser.parse_args()


def load_rows(paths):
    rows = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            rows.extend(json.load(handle))
    return rows


def main() -> None:
    args = parse_args()
    result_paths = args.results_json or [
        "artifacts/results/gpt4_family_results.json",
        "artifacts/results/gpt5_family_results.json",
    ]

    rows = load_rows(result_paths)
    fit = qhp.fit_scaling_law(rows)
    bootstrap = qhp.bootstrap_fit(rows, samples=args.bootstrap_samples, seed=args.seed)
    summary = qhp.summarize_results(rows)
    fit_payload = {
        "k": fit[0],
        "alpha": fit[1],
        "beta": fit[2],
        "equation": "d_phys = k * S^(-alpha) * D^(beta)",
        "fit_method": "log_linear_least_squares",
        "bootstrap": bootstrap,
        "is_partial": False,
    }

    qhp.save_json(args.combined_results_json, rows)
    qhp.save_json(args.summary_json, summary)
    qhp.save_json(args.fit_json, fit_payload)
    qhp.save_overview_plot(args.overview_plot)
    qhp.save_takeaway_plot(rows, args.takeaway_plot, False)
    qhp.save_main_plot(rows, fit, bootstrap, args.main_plot, False)
    qhp.save_breakdown_plot(rows, args.breakdown_plot, False)
    qhp.save_ablation_plot(rows, args.ablation_plot, False)

    cmd = [sys.executable, str(ROOT / "scripts" / "render_results_tex.py")]
    for path in result_paths:
        cmd.extend(["--results-json", path])
    cmd.extend(["--fit-json", args.fit_json, "--output-tex", args.output_tex])
    subprocess.run(cmd, check=True)

    print(f"Wrote {Path(args.combined_results_json).name}")
    print(f"Wrote {Path(args.summary_json).name}")
    print(f"Wrote {Path(args.fit_json).name}")
    print(f"Wrote {Path(args.overview_plot).name}")
    print(f"Wrote {Path(args.takeaway_plot).name}")
    print(f"Wrote {Path(args.main_plot).name}")
    print(f"Wrote {Path(args.breakdown_plot).name}")
    print(f"Wrote {Path(args.ablation_plot).name}")
    print(f"Wrote {Path(args.output_tex).name}")


if __name__ == "__main__":
    main()
