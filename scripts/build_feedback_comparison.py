#!/usr/bin/env python3

import argparse
import json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a compact feedback-ablation comparison JSON from three time-budget summaries."
    )
    parser.add_argument("--no-retry-summary", required=True)
    parser.add_argument("--self-retry-summary", required=True)
    parser.add_argument("--verifier-feedback-summary", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def load(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def group_rows(summary):
    grouped = {}
    for row in summary["rows"]:
        grouped[(row["model"], row["category"])] = row
    return grouped


def main() -> None:
    args = parse_args()
    no_retry = load(args.no_retry_summary)
    self_retry = load(args.self_retry_summary)
    verifier_feedback = load(args.verifier_feedback_summary)

    payload = {
        "overall": {
            "no_retry": {
                "solve_rate": no_retry["solve_rate"],
                "mean_solve_time_s": no_retry["mean_solve_time_s"],
                "median_solve_time_s": no_retry["median_solve_time_s"],
            },
            "self_retry": {
                "solve_rate": self_retry["solve_rate"],
                "mean_solve_time_s": self_retry["mean_solve_time_s"],
                "median_solve_time_s": self_retry["median_solve_time_s"],
            },
            "verifier_feedback": {
                "solve_rate": verifier_feedback["solve_rate"],
                "mean_solve_time_s": verifier_feedback["mean_solve_time_s"],
                "median_solve_time_s": verifier_feedback["median_solve_time_s"],
            },
        },
        "by_category": {},
        "by_model": {},
        "deltas": {
            "self_retry_vs_no_retry": self_retry["solve_rate"] - no_retry["solve_rate"],
            "verifier_feedback_vs_no_retry": verifier_feedback["solve_rate"] - no_retry["solve_rate"],
            "verifier_feedback_vs_self_retry": verifier_feedback["solve_rate"] - self_retry["solve_rate"],
        },
    }

    grouped = {
        "no_retry": group_rows(no_retry),
        "self_retry": group_rows(self_retry),
        "verifier_feedback": group_rows(verifier_feedback),
    }

    categories = sorted({category for _, category in grouped["verifier_feedback"].keys()})
    models = sorted({model for model, _ in grouped["verifier_feedback"].keys()})

    for category in categories:
        payload["by_category"][category] = {}
        for mode in ("no_retry", "self_retry", "verifier_feedback"):
            rows = [row for (model, cat), row in grouped[mode].items() if cat == category]
            payload["by_category"][category][mode] = sum(row["solve_rate"] for row in rows) / len(rows)

    for model in models:
        payload["by_model"][model] = {}
        for mode in ("no_retry", "self_retry", "verifier_feedback"):
            rows = [row for (name, category), row in grouped[mode].items() if name == model]
            payload["by_model"][model][mode] = sum(row["solve_rate"] for row in rows) / len(rows)

    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Saved comparison to {args.output_json}")


if __name__ == "__main__":
    main()
