from typing import TYPE_CHECKING, Optional

import pandas as pd

if TYPE_CHECKING:
    from icefall.stream_runtime.operator import PipelineOperator


class OperatorLink:
    """
    A link between two pipeline operators
    """

    def __init__(self, source: "PipelineOperator", target: "PipelineOperator") -> None:
        self.source = source
        self.target = target
        self.source.add_out_link(self)
        self.target.add_in_link(self)
        self.rows = None

    def send_rows(self, rows: pd.DataFrame) -> None:
        if self.rows is None:
            self.rows = rows.copy()
        else:
            self.rows = pd.concat([self.rows, rows], ignore_index=True)

    def get_rows(self, num_rows: Optional[int] = None) -> pd.DataFrame:
        if self.rows is None:
            raise ValueError("No rows to get")
        if num_rows is None or num_rows > len(self.rows):
            num_rows = len(self.rows)
        rows = self.rows.head(num_rows)
        self.rows = self.rows.drop(rows.index)
        return rows.reset_index(drop=True)

    def is_empty(self) -> bool:
        return self.rows is None or len(self.rows) == 0
