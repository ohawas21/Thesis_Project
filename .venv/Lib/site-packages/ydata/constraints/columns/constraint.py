from typing import Callable

from dask import compute, delayed
from numpy import count_nonzero as np_count_nonzero
from numpy import inf as np_inf
from numpy import isnan as np_isnan
from numpy import mean as np_mean
from numpy import quantile as np_quantile
from numpy import std as np_std
from numpy import sum as np_sum
from numpy import unique as np_unique
from pandas import DataFrame as pdDataFrame
from pandas import Interval as pdInterval

from ydata.constraints.base import ColumnConstraint
from ydata.dataset.dataset import Dataset


# @typechecked
class CustomConstraint(ColumnConstraint):
    _constraint_class: str = "CustomColumn"

    def __init__(
        self,
        check: Callable,
        columns: str | list[str] | None = None,
        name: str | None = None,
        available_columns:  list[str] | None = None,
        entity: str | None = None,
    ):

        self._columns = columns if columns is not None else []
        if isinstance(self._columns, str):
            self._columns = [self._columns]
        self._check = check
        self.name = name
        self.available_columns = (
            available_columns if available_columns is not None else []
        )
        self._entity = entity

    def validate(self, dataset: Dataset) -> pdDataFrame:
        from ydata.utils.sampling.proportion import calculate_wilson_sample_size
        sample_size = calculate_wilson_sample_size(1_000_000, 0.8, 2.58, 0.01)
        sample_size = int(sample_size)
        dataset = dataset.sample(sample_size)

        mask = {}
        for column in self._columns:
            mask[column] = self._check(dataset._data[column])

        return pdDataFrame(compute(mask))

    def __str__(self):
        if self.name:
            return f"{self._constraint_class} '{self.name}' on columns [{', '.join(self._columns)}]"
        return f"{self._constraint_class} on columns [{', '.join(self._columns)}]"

    def __repr__(self):
        return str(self)


# @typechecked
class Interval(CustomConstraint):
    _constraint_class: str = "IntervalColumn"

    def __init__(
        self,
        check: Callable,
        columns: str | list[str] | None = None,
        name: str | None = None,
        available_columns: list[str] | None = None,
        entity: str | None = None,
        lower_bound: int | float = -np_inf,
        upper_bound: int | float = np_inf,
        closed: str = "right",
    ):
        super().__init__(
            self._create_check(check, pdInterval(
                lower_bound, upper_bound, closed)),
            columns,
            name,
            available_columns,
            entity,
        )

    @staticmethod
    def _create_check(check: Callable, interval: pdInterval) -> Callable:
        @delayed
        def interval_check(data):
            return check(data) in interval

        return interval_check


# @typechecked
class GreaterThan(Interval):
    _constraint_class: str = "GreaterThanColumn"

    def __init__(
        self,
        check: Callable,
        value: int | float,
        columns: str | list[str] | None = None,
        name: str | None = None,
        available_columns: list[str] | None = None,
        entity: str | None = None,
    ):
        super().__init__(
            check, columns, name, available_columns, entity, lower_bound=value
        )


# @typechecked
class LowerThan(Interval):
    _constraint_class: str = "LowerThanColumn"

    def __init__(
        self,
        check: Callable,
        value: int | float,
        columns: str | list[str] | None = None,
        name: str | None = None,
        available_columns: list[str] | None = None,
        entity: str | None = None,
    ):
        super().__init__(
            check, columns, name, available_columns, entity, upper_bound=value
        )


# @typechecked
class Equal(Interval):
    _constraint_class: str = "EqualColumn"

    def __init__(
        self,
        check: Callable,
        value: int | float,
        tolerance: int | float = 0,
        columns: str | list[str] | None = None,
        name: str | None = None,
        available_columns: list[str] | None = None,
        entity: str | None = None,
    ):
        super().__init__(
            check,
            columns,
            name,
            available_columns,
            entity,
            lower_bound=value - tolerance,
            upper_bound=value + tolerance,
            closed="both",
        )


# @typechecked
class StandardDeviationBetween(Interval):
    _constraint_class: str = "StandardDeviationBetween"

    def __init__(
        self,
        lower_bound: int | float,
        upper_bound: int | float,
        columns: str | list[str] | None = None,
        name: str | None = None,
        available_columns: list[str] | None = None,
        entity: str | None = None,
    ):
        super().__init__(
            check=np_std,
            columns=columns,
            name=name,
            available_columns=available_columns,
            entity=entity,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )


# @typechecked
class MeanBetween(Interval):
    _constraint_class: str = "MeanBetweenRange"

    def __init__(
        self,
        lower_bound: int | float,
        upper_bound: int | float,
        columns: str | list[str] | None = None,
        name: str | None = None,
        available_columns: list[str] | None = None,
        entity: str | None = None,
    ):
        super().__init__(
            check=np_mean,
            columns=columns,
            name=name,
            available_columns=available_columns,
            entity=entity,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )


# @typechecked
class QuantileBetween(Interval):
    _constraint_class: str = "QuantileBetweenRange"

    def __init__(
        self,
        quantile: float,
        lower_bound: int | float,
        upper_bound: int | float,
        columns: str | list[str] | None = None,
        name: str | None = None,
        available_columns: list[str] | None = None,
        entity: str | None = None,
    ):
        self.quantile = quantile
        super().__init__(
            check=self.calculate_quantile,
            columns=columns,
            name=name,
            available_columns=available_columns,
            entity=entity,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

    def calculate_quantile(self, data):
        return np_quantile(data, self.quantile)


# @typechecked
class UniqueValuesBetween(Interval):
    _constraint_class: str = "UniqueValuesInRange"

    def __init__(
        self,
        lower_bound: int | float,
        upper_bound: int | float,
        columns: str | list[str] | None = None,
        name: str | None = None,
        available_columns: list[str] | None = None,
        entity: str | None = None,
    ):
        super().__init__(
            check=self.calculate_unique_counts,
            columns=columns,
            name=name,
            available_columns=available_columns,
            entity=entity,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

    def calculate_unique_counts(self, data):
        return len(np_unique(data))


# @typechecked
class Constant(Interval):
    _constraint_class: str = "Constant"

    def __init__(
        self,
        columns: str | list[str] | None = None,
        name: str | None = None,
        available_columns: list[str] | None = None,
        entity: str | None = None,
    ):
        super().__init__(
            check=self.calculate_unique_counts,
            columns=columns,
            name=name,
            available_columns=available_columns,
            entity=entity,
            lower_bound=0,
            upper_bound=1,
        )

    def calculate_unique_counts(self, data):
        return len(np_unique(data))


class NullValuesCountLowerThan(LowerThan):
    _constraint_class: str = "NullValuesCountLowerThan"

    def __init__(
        self,
        value: int | float,
        columns: str | list[str] | None = None,
        name: str | None = None,
        available_columns: list[str] | None = None,
        entity: str | None = None,
    ):
        super().__init__(
            check=self.count_null,
            value=value,
            columns=columns,
            name=name,
            available_columns=available_columns,
            entity=entity,
        )

    def count_null(self, data):
        return np_count_nonzero(np_isnan(data))


class SumLowerThan(LowerThan):
    _constraint_class: str = "SumLowerThan"

    def __init__(
        self,
        value: int | float,
        columns: str | list[str] | None = None,
        name: str | None = None,
        available_columns: list[str] | None = None,
        entity: str | None = None,
    ):
        super().__init__(
            check=np_sum,
            value=value,
            columns=columns,
            name=name,
            available_columns=available_columns,
            entity=entity,
        )
