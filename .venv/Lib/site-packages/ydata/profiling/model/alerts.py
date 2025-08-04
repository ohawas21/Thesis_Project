"""
    Custom function to calculate the alerts logic
"""

from typing import List, Dict, Optional

import numpy as np

from ydata_profiling.config import Settings
from ydata_profiling.model.alerts import (Alert, DuplicatesAlert, EmptyAlert, AlertType, alert_value,
                                          check_variable_alerts, check_correlation_alerts, fmt_percent)


class NearDuplicatesAlert(Alert):
    def __init__(
        self,
        values: Optional[Dict] = None,
        column_name: Optional[str] = None,
        is_empty: bool = False,
    ):
        super().__init__(
            alert_type=AlertType.NEAR_DUPLICATES,
            values=values,
            column_name=column_name,
            fields={"n_near_dups"},
            is_empty=is_empty,
        )

    def _get_description(self) -> str:
        if self.values is not None:
            return f"Dataset has {self.values['n_near_dups']} ({fmt_percent(self.values['p_near_dups'])}) near duplicate rows"
        else:
            return "Dataset has no near duplicated values"


def check_table_alerts(table: dict) -> List[Alert]:
    """Checks the overall dataset for alerts.

    Args:
        table: Overall dataset statistics.

    Returns:
        A list of alerts.
    """
    alerts: List[Alert] = []
    if alert_value(table.get("n_duplicates", np.nan)):
        alerts.append(
            DuplicatesAlert(
                values=table,
            )
        )

    if alert_value(table.get("n_near_dups", np.nan)):
        alerts.append(
            NearDuplicatesAlert(
                values=table,
            )
        )

    if table["n"] == 0:
        alerts.append(
            EmptyAlert(
                values=table,
            )
        )
    return alerts

def get_alerts(
    config: Settings, table_stats: dict, series_description: dict, correlations: dict
) -> List[Alert]:
    alerts: List[Alert] = check_table_alerts(table_stats)
    for col, description in series_description.items():
        alerts += check_variable_alerts(config, col, description)
    alerts += check_correlation_alerts(config, correlations)
    alerts.sort(key=lambda alert: str(alert.alert_type))
    return alerts
