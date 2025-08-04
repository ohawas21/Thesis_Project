from enum import Enum
from typing import Any


class MetricType(Enum):
    VISUAL = "visual"
    NUMERIC = "numeric"


class MetricScore:
    def __init__(
        self, name: str, type: MetricType, values: Any, description: str = ""
    ) -> None:
        self._name = name
        self._type = type
        self._values = values
        self._is_error = isinstance(values, Exception)
        self._description = description

    @property
    def name(self) -> str:
        return self._name

    @property
    def values(self) -> list:
        return self._values

    @property
    def type(self) -> MetricType:
        return self._type

    @property
    def description(self) -> str:
        return self._description

    @property
    def is_error(self) -> bool:
        return self._is_error
