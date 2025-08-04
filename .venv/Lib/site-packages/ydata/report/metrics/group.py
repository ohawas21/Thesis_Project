from ydata.report.logger import logger
from ydata.report.metrics.base_metric import BaseMetric
from ydata.report.metrics.score import MetricScore
from ydata.report.metrics.utils import is_entity


class MetricGroup:
    metrics: dict[str, BaseMetric]
    template: str

    def __init__(self, metrics: dict[str, BaseMetric], safe_mode: bool, template=None) -> None:
        self.metrics = metrics
        self.safe_mode = safe_mode
        self.template = template

    def evaluate(self, source, synthetic, **kwargs):
        result = {}

        meta = kwargs.get("metadata", None)
        for name, metric in self.metrics.items():
            if metric.exclude_entity_col:
                columns = [k for k in source.columns if not is_entity(k, meta)]
                if len(columns) == 0:
                    logger.warning(
                        "No column available after removing entity columns."
                        f"Metric {name} will not be calculated."
                    )
                    result[name] = {}
                    continue
                sou_ = source[columns].copy()
                syn = synthetic[columns].copy()
            else:
                sou_ = source.copy()
                syn = synthetic.copy()
            try:
                result[name] = metric.evaluate(sou_, syn, **kwargs)
            except Exception as e:
                if not self.safe_mode:
                    raise e
                logger.exception(e)
                result[name] = MetricScore(
                    name=metric.name,
                    type=metric.type,
                    values=e,
                    description=metric._description,
                )
        return result
