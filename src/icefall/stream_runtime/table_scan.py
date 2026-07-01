import pandas as pd

from icefall.stream_runtime.operator import PipelineOperator, PipelineOperatorWorker


class TableScanWorker(PipelineOperatorWorker):
    def finalize_construction(self, data: pd.DataFrame) -> None:
        self.data = data

    def compute(self, rows: pd.DataFrame, in_link_number: int) -> None:
        pass

    def terminate(self) -> None:
        for out_link in self.out_links:
            out_link.send_rows(self.data)


class TableScan(PipelineOperator[TableScanWorker]):
    dop = 1
