from enum import Enum
from typing import Any

class DatasetType(Enum):
    TABULAR = 'tabular'
    TIMESERIES = 'timeseries'
    @classmethod
    def get(cls, name: str, default: Any = None): ...
    @classmethod
    def list(cls): ...
