import time

from ydata.dataset import Dataset
from ydata.report.logger import logger
from ydata.report.metrics.score import MetricScore, MetricType


class BaseMetric:
    def __init__(self, formatter, exclude_entity_col: bool = True) -> None:
        self._description = self._get_description(formatter)
        self.exclude_entity_col: bool = exclude_entity_col

    @staticmethod
    def _get_description(formatter):
        return ""

    def _evaluate(self, source: Dataset, synthetic: Dataset, **kwargs) -> MetricScore:
        raise NotImplementedError

    def evaluate(self, source, synthetic, **kwargs) -> MetricScore:
        logger.info(f"[PROFILEREPORT] - Calculating metric [{self.name}].")
        start = time.perf_counter()
        values = self._evaluate(source, synthetic, **kwargs)
        logger.info(
            f"[PROFILEREPORT] - Metric [{self.name}] took {(time.perf_counter() - start):.2f}s."
        )
        return MetricScore(
            name=self.name,
            type=self.type,
            values=values,
            description=self._description,
        )

    @property
    def type(self) -> MetricType:
        return MetricType.NUMERIC

    @property
    def name(self) -> str:
        raise NotImplementedError
