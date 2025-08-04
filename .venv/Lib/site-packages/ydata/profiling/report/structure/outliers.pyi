from ydata_profiling.config import Settings as Settings
from ydata_profiling.report.presentation.core.renderable import Renderable as Renderable

def get_outliers_items(config: Settings, summary: list | dict) -> Renderable:
    """Create the list of outlier items.

    Args:
        config: report Settings object
        outliers: outliers data

    Returns:
        List of outliers information to show in the interface.
    """
