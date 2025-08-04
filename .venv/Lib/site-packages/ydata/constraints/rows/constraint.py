import re
from enum import Enum
from typing import Any, Callable

from pandas import DataFrame as pdDataFrame
from pandas import Interval as pdInterval
from pandas import Timedelta as pdTimedelta
from pandas import isna

from ydata.constraints.base import Axis, RowConstraint
from ydata.constraints.utils import check_row_constraint, check_series_relation
from ydata.dataset.dataset import Dataset


# @typechecked
class CustomConstraint(RowConstraint):
    _constraint_class: str = "CustomConstraint"

    def __init__(
        self,
        check: Callable,
        columns: str | list[str] | None = None,
        name: str | None = None,
        axis: Axis = "column",
        available_columns: list[str] | None = None,
        entity: str | None = None,
    ):
        """Check a row constraint.

        Args:
            check (Callable): Serie constraint function
            columns (str | list[str] | None): List of columns on which the constraint applies to
            name (str): User-friendly name for the constraint
            axis (Literal[CUSTOM_CONSTRAINT_AXIS]): Execution mode for the constraint. Default 'column',
            available_columns (list[str] | None): Columns made available to the constraint
            entity: (str | None): Column to consider as entity

        Returns:
            CustomConstraint: custom constraint instance
        """
        self._columns = columns if columns is not None else []
        self._columns = (
            [self._columns] if isinstance(
                self._columns, str) else self._columns
        )
        self._check = check
        self.axis = RowConstraint._get_axis_number(axis)
        self.name = name
        self.available_columns = (
            available_columns if available_columns is not None else []
        )
        self._entity = entity

        self._is_serie_constraint = self.__is_serie_constraint(
            self.available_columns)

    @staticmethod
    def __is_serie_constraint(columns) -> bool:
        """Deduce if a constraint is a serie constraint from a list of columns.

        Args:
            columns (list[str]): List of columns

        Returns:
            bool: True is the constraint is a serie constraint
        """
        return any("|n" in c for c in columns)

    def validate(self, dataset: Dataset) -> pdDataFrame:
        """Validate the constraint against a dataset.

        Args:
            dataset (Dataset): Dataset to test

        Returns:
            pdDataFrame: constraint mask
        """
        cols = self.available_columns + self._columns
        if self._entity:
            cols.append(self._entity)
        cols = list(set(cols))

        if self.axis in [1, "rows"]:
            d = dataset[cols] if len(cols) else dataset
            mask = check_row_constraint(d, self._check, self._entity)
        else:
            if self._is_serie_constraint:
                mask = check_series_relation(
                    dataset, self._check, self.available_columns
                )
            else:
                d = dataset[cols] if len(cols) else dataset
                # Because of how Dask works, if there is an aggregation in _check, it will return the wrong result
                # mask = self._check(d._data).compute()
                if self._entity:
                    mask = (
                        dataset._data.groupby(self._entity)
                        .apply(self._check, meta=(None, bool))
                        .compute()
                    )
                    mask = mask.reset_index(level=0, drop=True)
                else:
                    mask = self._check(d._data.compute())
        if self._columns:
            return mask[self._columns]
        else:
            return pdDataFrame(mask)

    def __str__(self):
        if self.name:
            return f"{self._constraint_class} '{self.name}' on columns [{', '.join(self._columns)}]"
        return f"{self._constraint_class} on columns [{', '.join(self._columns)}]"

    def __repr__(self):
        return str(self)


# @typechecked
class GreaterThan(RowConstraint):
    def __init__(self, columns: str | list[str], value: float | str):
        """Check if a set of columns are greater than a value or another
        column.

        Args:
            columns (str | list[str] | None): List of columns on which the constraint applies to
            value (float | str): value to consider. If value is a string, it must be a column.

        Returns:
            GreaterThan: GreaterThan constraint instance
        """
        self._columns: list = columns if isinstance(
            columns, list) else [columns]
        self._value = value

        if isinstance(self._value, str) and len(self._columns) > 1:
            raise AttributeError(
                "Argument columns must be a single column when value is a string"
            )

    def validate(self, dataset: Dataset) -> pdDataFrame:
        """Validate the constraint against a dataset.

         Args:
            dataset (Dataset): Dataset to test

        Returns:
            pdDataFrame: constraint mask
        """

        from ydata.utils.sampling.proportion import calculate_wilson_sample_size
        sample_size = calculate_wilson_sample_size(1_000_000, 0.8, 2.58, 0.01)
        sample_size = int(sample_size)
        dataset = dataset.sample(sample_size)

        if isinstance(self._value, str):
            mask = pdDataFrame(
                (
                    dataset._data[self._columns].ge(
                        dataset._data[self._value], axis=0)
                ).compute()
            )
        else:
            mask = (dataset._data[self._columns] > self._value).compute()
        return mask

    def __str__(self):
        return (
            f"GreaterThan {'column ' if isinstance(self._value, str) else ''}{self._value} "
            f"on columns [{self._columns if isinstance(self._value, str) else ', '.join(self._columns)}]"
        )

    def __repr__(self):
        return f"GreaterThan(columns={self._columns}, value={self._value})"


# @typechecked
class LowerThan(RowConstraint):
    def __init__(self, columns: str | list[str], value: float | str):
        """Check if a set of columns are greater than a value or another
        column.

        Args:
            columns (str | list[str] | None): List of columns on which the constraint applies to
            value (float | str): value to consider. If value is a string, it must be a column.

        Returns:
            LowerThanRowConstraint: LowerThanRowConstraint constraint instance
        """
        self._columns: list = columns if isinstance(
            columns, list) else [columns]
        self._value = value

        if isinstance(self._value, str) and len(self._columns) > 1:
            raise AttributeError(
                "Argument columns must be a single column when value is a string"
            )

    def validate(self, dataset: Dataset) -> pdDataFrame:
        """Validate the constraint against a dataset.

         Args:
            dataset (Dataset): Dataset to test

        Returns:
            pdDataFrame: constraint mask
        """
        if isinstance(self._value, str):
            mask = pdDataFrame(
                (
                    dataset._data[self._columns].le(
                        dataset._data[self._value], axis=0)
                ).compute()
            )
        else:
            mask = (dataset._data[self._columns] < self._value).compute()
        return mask

    def __str__(self):
        return (
            f"LowerThan {'column ' if isinstance(self._value, str) else ''}{self._value} "
            f"on columns [{self._columns if isinstance(self._value, str) else ', '.join(self._columns)}]"
        )

    def __repr__(self):
        return f"LowerThan(columns={self._columns}, value={self._value})"


# @typechecked
class Between(RowConstraint):
    def __init__(
        self,
        columns: str | list[str],
        lower_bound: int | float,
        upper_bound: int | float,
    ):
        """Check if a set of columns are greater than a value or another
        column.

        Args:
            columns (str | list[str] | None): List of columns on which the constraint applies to
            value (float | str): value to consider. If value is a string, it must be a column.

        Returns:
            BetweenRowConstraint: BetweenRowConstraint constraint instance
        """
        self._columns: list = columns if isinstance(
            columns, list) else [columns]
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

    def validate(self, dataset: Dataset) -> pdDataFrame:
        """Validate the constraint against a dataset.

         Args:
            dataset (Dataset): Dataset to test

        Returns:
            pdDataFrame: constraint mask
        """

        mask = (dataset._data[self._columns] >= self._lower_bound) & (
            dataset._data[self._columns] <= self._upper_bound
        )
        return mask.compute()

    def __str__(self):
        if isinstance(self._columns, str):
            cols = [str]
        else:
            cols = self._columns
        return (
            f"Between {self._lower_bound} and {self._upper_bound} "
            f"on columns [{cols[0] if len(cols)==1 else ', '.join(self._columns)}]"
        )

    def __repr__(self):
        return f"Between(columns={self._columns}, value={self._lower_bound, self._upper_bound})"


# @typechecked
class Positive(GreaterThan):
    def __init__(self, columns: str | list[str]):
        """Check if a set of columns are greater than 0.

        Args:
            columns (str | list[str] | None): List of columns on which the constraint applies to
            value (float | str): value to consider. If value is a string, it must be a column.

        Returns:
            Positive: Positive constraint instance
        """
        super().__init__(columns, value=0)

    def validate(self, dataset: Dataset) -> pdDataFrame:
        """Validate the constraint against a dataset.

        Args:
            dataset (Dataset): Dataset to test

        Returns:
            pdDataFrame: constraint mask
        """
        return super().validate(dataset)

    def __str__(self):
        return f"Positive on columns [{', '.join(self._columns)}]"

    def __repr__(self):
        return f"Positive(columns={self._columns})"


# @typechecked
class BetweenDates(RowConstraint):
    def __init__(self, constrained_column: str, reference_column: str, lower_bound: int, upper_bound: int):
        """Check if a date column is between an interval of days with reference
        to another date column.

        Args:
            constrained_column (str): Date column on which the constraint is applied.
            reference_column (str): Date column which serves as a reference for the constraint.
            lower_bound (int): Lower bound of the interval in days.
            upper_bound (int): Upper bound of the interval in days.

        Returns:
            BetweenDates: BetweenDates constraint instance
        """
        self._constrained_column = constrained_column
        self._reference_column = reference_column
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

    def _check_within_interval(self, row) -> bool:
        return row[self._constrained_column] in pdInterval(
            row[self._reference_column] +
            pdTimedelta(self._lower_bound, "day"),
            row[self._reference_column] +
            pdTimedelta(self._upper_bound, "day"),
            closed="both"
        )

    def validate(self, dataset: Dataset) -> pdDataFrame:
        """Validate the constraint against a dataset.

         Args:
            dataset (Dataset): Dataset to test

        Returns:
            pdDataFrame: constraint mask
        """
        mask = dataset.to_dask().apply(self._check_within_interval,
                                       axis=1, meta=(None, 'bool'))
        return pdDataFrame(mask, columns=[self._constrained_column])

    def __str__(self):
        return (
            f"Date column {self._constrained_column} is between {self._lower_bound} "
            f"and {self._upper_bound} days with reference to date column {self._reference_column}"
        )

    def __repr__(self):
        return f"BetweenDates(\
            constrained_column={self._constrained_column}, \
            reference_column={self._reference_column}, \
            lower_bound={self._lower_bound}, \
            upper_bound={self._upper_bound})"


# @typechecked
class IncludedIn(RowConstraint):
    def __init__(self, column: str, values: list[Any] | Any):
        """Check if the column values are included in a specified list of
        values.

        Args:
            column (str): Column on which the constraint is applied.
            values (list): List of specified values (or a single value).

        Returns:
            IncludedIn: IncludedIn constraint instance
        """
        self._column = column
        self._values = values if isinstance(values, list) else [values]

    def _check_if_value_exists(self, row) -> bool:
        if isna(row[self._column]):
            return False
        return row[self._column] in self._values

    def validate(self, dataset: Dataset) -> pdDataFrame:
        """Validate the constraint against a dataset.

         Args:
            dataset (Dataset): Dataset to test

        Returns:
            pdDataFrame: constraint mask
        """
        mask = dataset.to_dask().apply(self._check_if_value_exists,
                                       axis=1, meta=(None, 'bool'))
        return pdDataFrame(mask, columns=[self._column])

    def __str__(self):
        return (
            f"Column {self._column} values are included in the list [{', '.join(self._values)}]"
        )

    def __repr__(self):
        return f"IncludedIn(column={self._column}, values={self._values})"


# @typechecked
class Regex(RowConstraint):
    def __init__(self, column: str, regex: str):
        """Check if the column values match a regular expression.

        Args:
            column (str): Column on which the constraint is applied.
            regex (str): Regular expression.

        Returns:
            Regex: Regex constraint instance
        """
        self._column = column
        self._regex = regex

    def validate(self, dataset: Dataset) -> pdDataFrame:
        """Validate the constraint against a dataset.

         Args:
            dataset (Dataset): Dataset to test

        Returns:
            pdDataFrame: constraint mask
        """
        mask = dataset.to_dask().apply(lambda row: bool(re.fullmatch(
            self._regex, str(row[self._column]))), axis=1, meta=(None, 'bool'))
        return pdDataFrame(mask, columns=[self._column])

    def __str__(self):
        return (
            f"Column {self._column} values match the regular expression '{self._regex}'"
        )

    def __repr__(self):
        return f"Regex(column={self._column}, regex={self._regex})"


# @typechecked
class CombineConstraints(RowConstraint):
    class Operation(Enum):
        MERGE = "AND"
        CHAIN = "XAND"

    def __init__(self, constraints: list[RowConstraint], operation: Operation = Operation.CHAIN):
        """Combines a list of constraints.

        Args:
            constraints (str): List of constraints.
            operation (Operation): Operation that specifies how to combine. Defaults to CHAIN.

        Returns:
            CombinedConstraints: CombinedConstraints constraint instance
        """
        self._constraints = constraints
        self._operation = operation

    def validate(self, dataset: Dataset) -> pdDataFrame:
        """Validate the constraint against a dataset.

         Args:
            dataset (Dataset): Dataset to test

        Returns:
            pdDataFrame: constraint mask
        """
        all_masks = pdDataFrame()
        for i, constraint in enumerate(self._constraints):
            const_mask = constraint.validate(dataset)
            const_mask = const_mask.astype(int).sum(
                axis=1) == len(const_mask.columns)
            all_masks[f"mask_{i}"] = const_mask

        comb_masks = all_masks.sum(axis=1)
        if self._operation == self.Operation.MERGE:
            comb_masks = comb_masks == len(all_masks.columns)
        elif self._operation == self.Operation.CHAIN:
            comb_masks = (comb_masks == len(
                all_masks.columns)) | (comb_masks == 0)

        return pdDataFrame(comb_masks, columns=["combine_constraints"])

    def __str__(self):
        return (
            f"Constraints {[', '.join([str(c) for c in self._constraints])]} are combined using the {self._operation.name} operation"
        )

    def __repr__(self):
        return f"CombineConstraints(constraints={self._constraints}, operation={self._operation})"
