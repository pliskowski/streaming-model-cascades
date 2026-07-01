import abc
import random
from typing import Any, Generic, Optional, TypeVar, final, get_args

import pandas as pd

from icefall.stream_runtime.operator_link import OperatorLink


class PipelineOperatorWorker(abc.ABC):
    """Worker interface for a streaming pipeline operator."""

    @final
    def __init__(self) -> None:
        self.in_links: list[OperatorLink] = []
        self.out_links: list[OperatorLink] = []

    @abc.abstractmethod
    def finalize_construction(self, *args, **kwargs) -> None:
        """
        Finalizes operator worker construction.
        """
        pass

    @abc.abstractmethod
    def compute(self, rows: pd.DataFrame, in_link_number: int) -> None:
        pass

    @abc.abstractmethod
    def terminate(self) -> None:
        pass

    @final
    def add_in_link(self, in_link: OperatorLink) -> None:
        """
        Adds an inbound link to the operator.
        """
        self.in_links.append(in_link)

    @final
    def add_out_link(self, out_link: OperatorLink) -> None:
        """
        Adds an outbound link to the operator.
        """
        self.out_links.append(out_link)


T = TypeVar("T")


class MyGenericClass(Generic[T]):
    _type_T: Any

    def __init_subclass__(cls) -> None:
        cls._type_T = get_args(cls.__orig_bases__[0])[0]  # type: ignore


V = TypeVar("V", bound=PipelineOperatorWorker)


class PipelineOperator(MyGenericClass[V]):
    min_rows: int = 0
    max_rows: Optional[int] = None
    dop: int

    def __init__(self, *args, **kwargs) -> None:
        self.in_links: list[OperatorLink] = []
        self.out_links: list[OperatorLink] = []
        self.workers = [self._type_T() for _ in range(self.dop)]
        for worker in self.workers:
            worker.finalize_construction(*args, **kwargs)
        self.next_worker = 0

    def execute(self) -> None:
        while self.in_links and not all(link.is_empty() for link in self.in_links):
            for idx, in_link in enumerate(self.in_links):
                if in_link.is_empty():
                    continue
                num_rows = None
                if self.max_rows is not None:
                    num_rows = random.randint(self.min_rows, self.max_rows)
                in_rows = in_link.get_rows(num_rows)
                self.workers[self.next_worker].compute(in_rows, idx)
                self.next_worker = (self.next_worker + 1) % self.dop

        for worker in self.workers:
            worker.terminate()

    def add_in_link(self, in_link: OperatorLink) -> None:
        """
        Adds an inbound link to the operator.
        """
        self.in_links.append(in_link)
        for worker in self.workers:
            worker.add_in_link(in_link)

    def add_out_link(self, out_link: OperatorLink) -> None:
        """
        Adds an outbound link to the operator.
        """
        self.out_links.append(out_link)
        for worker in self.workers:
            worker.add_out_link(out_link)
