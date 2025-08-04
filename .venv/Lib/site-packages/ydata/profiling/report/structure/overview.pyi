from ydata_profiling.config import Settings as Settings
from ydata_profiling.model import BaseDescription as BaseDescription
from ydata_profiling.report.presentation.core.renderable import Renderable as Renderable

def get_dataset_items(config: Settings, summary: BaseDescription, alerts: list) -> list:
    """Returns the dataset overview (at the top of the report)

    Args:
        config: settings object
        summary: the calculated summary
        alerts: the alerts

    Returns:
        A list with components for the dataset overview (overview, reproduction, alerts)
    """
