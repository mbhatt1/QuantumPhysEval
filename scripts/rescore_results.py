#!/usr/bin/env python3

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quantumphyseval import benchmark as qhp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rescore saved benchmark outputs with the current parser and regenerate summary artifacts."
    )
    parser.add_argument(
        "--results-json",
        required=True,
        help="Input raw results JSON with saved model outputs.",
    )
    parser.add_argument(
        "--output-results-json",
        default=None,
        help="Output path for rescored rows. Defaults to overwriting --results-json.",
    )
    parser.add_argument(
        "--summary-json",
        required=True,
        help="Path to the rescored summary JSON.",
    )
    parser.add_argument(
        "--fit-json",
        required=True,
        help="Path to the rescored fit JSON.",
    )
    parser.add_argument(
        "--main-plot",
        required=True,
        help="Path to the rescored main figure.",
    )
    parser.add_argument(
        "--breakdown-plot",
        required=True,
        help="Path to the rescored breakdown figure.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=500,
        help="Bootstrap samples for the fit summary.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for bootstrap resampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_results_json = args.output_results_json or args.results_json

    with open(args.results_json, "r", encoding="utf-8") as handle:
        rows = json.load(handle)

    rescored = []
    for row in rows:
        updated = dict(row)
        target_obj = row.get("target")
        if not isinstance(target_obj, dict) or "kind" not in target_obj or "value" not in target_obj:
            raise ValueError("Each row must contain a serialized target with kind/value.")
        updated["d_phys"] = qhp.evaluate_output(
            row["category"],
            target_obj["kind"],
            target_obj["value"],
            row["output"],
        )
        rescored.append(updated)

    fit_params = qhp.fit_scaling_law(rescored)
    bootstrap = qhp.bootstrap_fit(rescored, samples=args.bootstrap_samples, seed=args.seed)
    summary = qhp.summarize_results(rescored)
    fit_payload = {
        "k": fit_params[0],
        "alpha": fit_params[1],
        "beta": fit_params[2],
        "equation": "d_phys = k * S^(-alpha) * D^(beta)",
        "fit_method": "log_linear_least_squares",
        "bootstrap": bootstrap,
        "is_partial": False,
    }

    qhp.save_json(output_results_json, rescored)
    qhp.save_json(args.summary_json, summary)
    qhp.save_json(args.fit_json, fit_payload)
    qhp.save_main_plot(rescored, fit_params, bootstrap, args.main_plot, False)
    qhp.save_breakdown_plot(rescored, args.breakdown_plot, False)

    print(f"Saved rescored results to {output_results_json}")
    print(f"Saved summary to {args.summary_json}")
    print(f"Saved fit to {args.fit_json}")
    print(f"Saved main plot to {args.main_plot}")
    print(f"Saved breakdown plot to {args.breakdown_plot}")


if __name__ == "__main__":
    main()
