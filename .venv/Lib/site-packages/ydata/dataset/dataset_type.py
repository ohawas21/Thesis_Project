from enum import Enum
from typing import Any


class DatasetType(Enum):
    TABULAR = "tabular"
    TIMESERIES = "timeseries"

    @classmethod
    def get(cls, name: str, default: Any = None):
        if name.lower() in cls.list():
            return cls[name.upper()]
        return default

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))
