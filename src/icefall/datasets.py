"""Dataset registry for cascade paper experiments.

Usage:
    from icefall.datasets import get_datasets, load_dataset

    datasets = get_datasets("curated")
    df = load_dataset("mmlu")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

_PROJECT_DIR = Path(__file__).resolve().parents[2]
_LOCAL_DATA_DIR = _PROJECT_DIR / "data"

PROXY_SCORE_COL = "proxy_pred_true_prob"
ORACLE_RESULT_COL = "oracle_pred"
INDEX_COL = "row_idx"


@dataclass
class DatasetSpec:
    """Specification for a single dataset."""

    name: str
    description: str
    merged_path: Optional[Path] = None

    def __post_init__(self):
        if self.merged_path is None:
            raise ValueError(f"Dataset {self.name}: must provide merged_path")


ALL_DATASETS: dict[str, DatasetSpec] = {
    "arxiv": DatasetSpec(
        name="arxiv",
        description="ArXiv paper abstracts, CS / stat / physics / math subject classification (56K rows)",
        merged_path=_LOCAL_DATA_DIR / "ARXIV_56180.csv",
    ),
    "boolq": DatasetSpec(
        name="boolq",
        description="BoolQ reading comprehension, 12.7K questions",
        merged_path=_LOCAL_DATA_DIR / "BOOLQ_P0_12697.csv",
    ),
    "imdb": DatasetSpec(
        name="imdb",
        description="IMDB sentiment, 50K reviews",
        merged_path=_LOCAL_DATA_DIR / "IMDB_50000.csv",
    ),
    "mmlu": DatasetSpec(
        name="mmlu",
        description="MMLU knowledge QA, 5K questions",
        merged_path=_LOCAL_DATA_DIR / "MMLU_5K.csv",
    ),
    "nyt": DatasetSpec(
        name="nyt",
        description="NYT article topic classification, 250K title/excerpt pairs (file name NYT_500K is historical; see data/README.md)",
        merged_path=_LOCAL_DATA_DIR / "NYT_500K.csv",
    ),
    "sst2": DatasetSpec(
        name="sst2",
        description="SST-2 sentiment analysis, 68K sentences",
        merged_path=_LOCAL_DATA_DIR / "SST2_68221.csv",
    ),
}

CURATED_DATASET_NAMES: list[str] = [
    "arxiv",
    "boolq",
    "imdb",
    "mmlu",
    "nyt",
    "sst2",
]


def _parse_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    s = str(val).strip().lower()
    return s in ("true", "1", "yes", "positive")


def _load_merged(path: Path) -> pd.DataFrame:
    """Load a pre-merged results CSV and normalize column names."""
    df = pd.read_csv(path)
    if PROXY_SCORE_COL not in df.columns:
        raise ValueError(f"Missing column '{PROXY_SCORE_COL}' in {path}")
    if ORACLE_RESULT_COL not in df.columns:
        raise ValueError(f"Missing column '{ORACLE_RESULT_COL}' in {path}")

    df[ORACLE_RESULT_COL] = df[ORACLE_RESULT_COL].apply(_parse_bool)

    cols = [PROXY_SCORE_COL, ORACLE_RESULT_COL]
    if INDEX_COL in df.columns:
        cols = [INDEX_COL] + cols
    return df[cols].copy()


def load_dataset(name: str) -> pd.DataFrame:
    """Load a single dataset as a merged DataFrame.

    Returns a DataFrame with columns:
      - proxy_pred_true_prob (float): proxy model's P(true)
      - oracle_pred (bool): oracle model's prediction. Coerced from the
        on-disk encoding (`true`/`false` strings for the shipped CSVs)
        by `_parse_bool` at load time.
      - row_idx (int): optional stable index, included as the leading
        column when the CSV ships it. The six curated CSVs in
        `data/` do not ship this column today (see data/README.md).
    """
    if name not in ALL_DATASETS:
        available = ", ".join(sorted(ALL_DATASETS.keys()))
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")

    return _load_merged(ALL_DATASETS[name].merged_path)


def get_datasets(selector: str = "curated") -> dict[str, DatasetSpec]:
    """Get a dictionary of datasets based on a selector.

    Args:
        selector: One of:
          - "curated": the hand-picked paper subset
          - "all": every available dataset
          - "name1,name2,...": comma-separated dataset names
    """
    if selector == "curated":
        return {name: ALL_DATASETS[name] for name in CURATED_DATASET_NAMES}
    elif selector == "all":
        return dict(ALL_DATASETS)
    else:
        names = [n.strip() for n in selector.split(",")]
        missing = [n for n in names if n not in ALL_DATASETS]
        if missing:
            available = ", ".join(sorted(ALL_DATASETS.keys()))
            raise ValueError(f"Unknown datasets: {missing}. Available: {available}")
        return {n: ALL_DATASETS[n] for n in names}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dataset registry info")
    parser.add_argument("--list", action="store_true", help="List all datasets")
    parser.add_argument("--check", action="store_true", help="Check all datasets can be loaded")
    parser.add_argument("--load", type=str, help="Load and print info for a dataset")
    args = parser.parse_args()

    if args.list:
        print(f"{'Name':<20} {'Description'}")
        for name, spec in sorted(ALL_DATASETS.items()):
            print(f"{name:<20} {spec.description}")
        print(f"\nCurated: {', '.join(CURATED_DATASET_NAMES)}")

    if args.check:
        for name in sorted(ALL_DATASETS.keys()):
            try:
                df = load_dataset(name)
                print(f"  OK  {name:<20} {len(df):>8} rows")
            except Exception as e:
                print(f"  FAIL {name:<20} {e}")

    if args.load:
        df = load_dataset(args.load)
        print(f"Dataset: {args.load}")
        print(f"Rows: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nHead:\n{df.head()}")
        print(f"\nProxy score stats:\n{df[PROXY_SCORE_COL].describe()}")
        print(f"\nOracle positive rate: {df[ORACLE_RESULT_COL].mean():.3f}")
