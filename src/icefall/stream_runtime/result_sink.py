from typing import Optional

import pandas as pd

from icefall.stream_runtime.operator import PipelineOperator, PipelineOperatorWorker


class ResultSinkWorker(PipelineOperatorWorker):
    def finalize_construction(self) -> None:
        self.data: dict[int, pd.DataFrame] = {}
        self.final_data: Optional[pd.DataFrame] = None

    def compute(self, rows: pd.DataFrame, in_link_number: int) -> None:
        if in_link_number not in self.data:
            self.data[in_link_number] = rows
        else:
            self.data[in_link_number] = pd.concat(
                [self.data[in_link_number], rows], ignore_index=True
            )

    def terminate(self) -> None:
        self.final_data = pd.concat(list(self.data.values()), axis=1)


class ResultSink(PipelineOperator[ResultSinkWorker]):
    dop = 1

    def get_result(self) -> pd.DataFrame:
        return self.workers[0].final_data
