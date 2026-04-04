# QuantumPhysEval

QuantumPhysEval is a reproducible benchmark for measuring **physics-grounded quantum hallucinations** in language models on exact two-qubit tasks.

It evaluates four task families:
- circuit evolution
- operator composition
- measurement prediction
- entanglement classification

Every prompt is self-labeled from exact quantum mechanics, so the benchmark scores model outputs against analytic ground truth rather than heuristic judges.

## Repo layout

- `quantumphyseval/`: installable benchmark package and CLI
- `scripts/`: helpers for pooled snapshots and LaTeX appendix generation
- `paper/`: manuscript sources
- `configs/`: documented benchmark presets
- `artifacts/`: generated results, figures, and paper outputs

## Install

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
```

You will also need:

```bash
export OPENAI_API_KEY=...
```

## Quick start

Run a small local benchmark:

```bash
python -m quantumphyseval.benchmark \
  --n-per-category 4 \
  --model gpt-4.1:1e10 \
  --reasoning-depths 1 2 4
```

Outputs are written to:

- `artifacts/results/`
- `artifacts/figures/`

## Reproduce the main benchmark slices

GPT-4 family:

```bash
python -m quantumphyseval.benchmark \
  --n-per-category 64 \
  --bootstrap-samples 2000 \
  --model gpt-4o-mini:2e9 \
  --model gpt-4.1-mini:5e9 \
  --model gpt-4.1:1e10 \
  --results-json artifacts/results/gpt4_family_results.json \
  --summary-json artifacts/results/gpt4_family_summary.json \
  --fit-json artifacts/results/gpt4_family_fit.json \
  --main-plot artifacts/figures/gpt4_family_scaling.png \
  --breakdown-plot artifacts/figures/gpt4_family_breakdown.png \
  --max-workers 32 \
  --checkpoint-every 50 \
  --plot-every 200
```

GPT-5 family:

```bash
python -m quantumphyseval.benchmark \
  --n-per-category 64 \
  --bootstrap-samples 2000 \
  --model gpt-5.1:1.1e10 \
  --model gpt-5.2:1.2e10 \
  --model gpt-5.4-mini:7e9 \
  --model gpt-5.4:1.4e10 \
  --results-json artifacts/results/gpt5_family_results.json \
  --summary-json artifacts/results/gpt5_family_summary.json \
  --fit-json artifacts/results/gpt5_family_fit.json \
  --main-plot artifacts/figures/gpt5_family_scaling.png \
  --breakdown-plot artifacts/figures/gpt5_family_breakdown.png \
  --max-workers 32 \
  --checkpoint-every 100 \
  --plot-every 400 \
  --validate-models \
  --skip-unavailable-models
```

Anthropic Claude family:

```bash
python -m quantumphyseval.benchmark \
  --provider anthropic \
  --n-per-category 64 \
  --bootstrap-samples 2000 \
  --model claude-sonnet-4-20250514:1.2e10 \
  --model claude-opus-4-1-20250805:1.6e10 \
  --results-json artifacts/results/anthropic_family_results.json \
  --summary-json artifacts/results/anthropic_family_summary.json \
  --fit-json artifacts/results/anthropic_family_fit.json \
  --main-plot artifacts/figures/anthropic_family_scaling.png \
  --breakdown-plot artifacts/figures/anthropic_family_breakdown.png \
  --max-workers 32 \
  --checkpoint-every 100 \
  --plot-every 400 \
  --validate-models \
  --skip-unavailable-models
```

If you improve the parser or output-extraction logic after a run has already completed, you can rescore saved raw outputs without paying for another API sweep:

```bash
python scripts/rescore_results.py \
  --results-json artifacts/results/anthropic_family_results.json \
  --summary-json artifacts/results/anthropic_family_summary.json \
  --fit-json artifacts/results/anthropic_family_fit.json \
  --main-plot artifacts/figures/anthropic_family_scaling.png \
  --breakdown-plot artifacts/figures/anthropic_family_breakdown.png
```

## Reproduce the feedback ablation

```bash
python -m quantumphyseval.benchmark \
  --experiment time_budget \
  --repair-mode verifier_feedback \
  --n-per-category 4 \
  --reasoning-depths 4 \
  --temperature 0.2 \
  --time-budget-seconds 300 \
  --checkpoint-seconds 30 60 120 300 \
  --max-attempts 24 \
  --model gpt-4.1:1e10 \
  --model gpt-5.2:1.2e10 \
  --model gpt-5.4-mini:7e9 \
  --model gpt-5.4:1.4e10 \
  --results-json artifacts/results/timebudget_strong_models_results.json \
  --summary-json artifacts/results/timebudget_strong_models_summary.json \
  --fit-json artifacts/results/timebudget_strong_models_meta.json \
  --main-plot artifacts/figures/timebudget_strong_models_progress.png \
  --breakdown-plot artifacts/figures/timebudget_strong_models_breakdown.png \
  --max-workers 32 \
  --validate-models \
  --skip-unavailable-models
```

## Build pooled artifacts and paper appendix

```bash
python scripts/build_pooled_snapshot.py \
  --results-json artifacts/results/gpt4_family_results.json \
  --results-json artifacts/results/gpt5_family_results.json \
  --results-json artifacts/results/anthropic_family_results.json
```

## Build the paper

```bash
cd paper
latexmk -pdf -interaction=nonstopmode QuantumPhysEval.tex
```

## Repro targets

Convenience targets are available via:

```bash
make help
```
