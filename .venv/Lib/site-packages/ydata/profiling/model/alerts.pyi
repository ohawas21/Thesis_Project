from ydata_profiling.config import Settings as Settings
from ydata_profiling.model.alerts import Alert

class NearDuplicatesAlert(Alert):
    def __init__(self, values: dict | None = None, column_name: str | None = None, is_empty: bool = False) -> None: ...

def check_table_alerts(table: dict) -> list[Alert]:
    """Checks the overall dataset for alerts.

    Args:
        table: Overall dataset statistics.

    Returns:
        A list of alerts.
    """
def get_alerts(config: Settings, table_stats: dict, series_description: dict, correlations: dict) -> list[Alert]: ...
