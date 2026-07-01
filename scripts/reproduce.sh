#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[1/3] Table 1: dataset characterization"
uv run python -m icefall.experiments.dataset_characterization

echo "[2/3] Exp 1: Pareto sweep (Tables 2–4, Figure 2)"
uv run python -m icefall.experiments.exp1_pareto --n-seeds 10

echo "[3/3] Exp 3: reliability grid (Table 5) — 14,450 runs"
uv run python -m icefall.experiments.exp3_reliability \
  --n-seeds 10 --fine-grid \
  --datasets arxiv,imdb,mmlu,boolq,sst2

echo "Generating publication tables..."
uv run python -m icefall.figures.tables \
  --input results/exp1_pareto/batch-4096_dop-1/exp1_pareto.csv \
  --experiment all

echo "Generating Pareto figure..."
uv run python -m icefall.figures.plot_pareto_pub --layout 2x3

echo "Done. Outputs in results/"
