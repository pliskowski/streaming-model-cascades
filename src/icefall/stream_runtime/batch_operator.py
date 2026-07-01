import copy

import pandas as pd

from icefall.stream_runtime.expr import RowExpr
from icefall.stream_runtime.operator import PipelineOperator, PipelineOperatorWorker


class BatchOperatorWorker(PipelineOperatorWorker):
    def finalize_construction(self, expr: RowExpr) -> None:
        self.expr = copy.deepcopy(expr)

    def compute(self, rows: pd.DataFrame, in_link_number: int) -> None:
        if in_link_number != 0:
            raise ValueError("in_link_number must be 0")
        out_rows = self.expr.compute(rows)
        self.out_links[0].send_rows(out_rows)

    def terminate(self) -> None:
        pass


class BatchOperator(PipelineOperator[BatchOperatorWorker]):
    min_rows = 256
    max_rows = 4096
    dop = 8
