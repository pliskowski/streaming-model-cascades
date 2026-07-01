from typing import Protocol

import pandas as pd


class RowExpr(Protocol):
    """Row expression protocol for streaming batch execution."""

    def compute(self, rows: pd.DataFrame) -> pd.DataFrame:
        pass
