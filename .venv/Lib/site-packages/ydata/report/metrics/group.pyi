from _typeshed import Incomplete
from ydata.report.metrics.base_metric import BaseMetric

class MetricGroup:
    metrics: dict[str, BaseMetric]
    template: str
    safe_mode: Incomplete
    def __init__(self, metrics: dict[str, BaseMetric], safe_mode: bool, template: Incomplete | None = None) -> None: ...
    def evaluate(self, source, synthetic, **kwargs): ...
