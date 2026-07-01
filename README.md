# Streaming Model Cascades for Semantic SQL

Artifact code for [*Streaming Model Cascades for Semantic SQL*](https://arxiv.org/abs/2604.00660) (Liskowski and Schmaus, 2026).

## What this repo contains

- Implementations of **SUPG-IT** and **GAMCAL** (+ baselines)
- Six benchmarks as precomputed proxy/oracle score CSVs (no raw source text)
- Scripts to reproduce the paper's tables and Pareto figure

The installable Python package is named `icefall` (e.g. `python -m icefall.experiments.exp1_pareto`).

Experiments replay pre-scored outputs from Llama 3.1-8B (proxy) and Llama 3.3-70B (oracle).

GAMCAL uses the `naive_ci` calibration path with uniform per-record quantiles and GAM confidence intervals in probability space — the implementation that produced the paper's experimental results.

## Quick start

```bash
uv sync --frozen --extra dev
uv run pytest -q tests/
```

Smoke test on one dataset:

```bash
uv run python -m icefall.experiments.exp1_pareto --datasets mmlu --n-seeds 2
```

## Full reproduction

```bash
bash scripts/reproduce.sh
```

This runs dataset characterization (Table 1), the Pareto sweep (Tables 2–4, Figure 2), and the reliability grid (Table 5). Expect several hours on a laptop.

## Citation

Paper: [https://arxiv.org/abs/2604.00660](https://arxiv.org/abs/2604.00660)

```bibtex
@misc{liskowski2026streaming,
  title         = {Streaming Model Cascades for Semantic SQL},
  author        = {Liskowski, Pawe{\l} and Schmaus, Kyle},
  year          = {2026},
  eprint        = {2604.00660},
  archivePrefix = {arXiv},
  primaryClass  = {cs.DB},
  url           = {https://arxiv.org/abs/2604.00660}
}
```

## License

MIT. See [LICENSE](LICENSE).
