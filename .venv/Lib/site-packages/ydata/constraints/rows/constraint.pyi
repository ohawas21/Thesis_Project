from _typeshed import Incomplete
from enum import Enum
from pandas import DataFrame as pdDataFrame
from typing import Any, Callable
from ydata.constraints.base import Axis as Axis, RowConstraint
from ydata.dataset.dataset import Dataset

class CustomConstraint(RowConstraint):
    axis: Incomplete
    name: Incomplete
    available_columns: Incomplete
    def __init__(self, check: Callable, columns: str | list[str] | None = None, name: str | None = None, axis: Axis = 'column', available_columns: list[str] | None = None, entity: str | None = None) -> None:
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
    def validate(self, dataset: Dataset) -> pdDataFrame:
        """Validate the constraint against a dataset.

        Args:
            dataset (Dataset): Dataset to test

        Returns:
            pdDataFrame: constraint mask
        """

class GreaterThan(RowConstraint):
    def __init__(self, columns: str | list[str], value: float | str) -> None:
        """Check if a set of columns are greater than a value or another
        column.

        Args:
            columns (str | list[str] | None): List of columns on which the constraint applies to
            value (float | str): value to consider. If value is a string, it must be a column.

        Returns:
            GreaterThan: GreaterThan constraint instance
        """
    def validate(self, dataset: Dataset) -> pdDataFrame:
        """Validate the constraint against a dataset.

         Args:
            dataset (Dataset): Dataset to test

        Returns:
            pdDataFrame: constraint mask
        """

class LowerThan(RowConstraint):
    def __init__(self, columns: str | list[str], value: float | str) -> None:
        """Check if a set of columns are greater than a value or another
        column.

        Args:
            columns (str | list[str] | None): List of columns on which the constraint applies to
            value (float | str): value to consider. If value is a string, it must be a column.

        Returns:
            LowerThanRowConstraint: LowerThanRowConstraint constraint instance
        """
    def validate(self, dataset: Dataset) -> pdDataFrame:
        """Validate the constraint against a dataset.

         Args:
            dataset (Dataset): Dataset to test

        Returns:
            pdDataFrame: constraint mask
        """

class Between(RowConstraint):
    def __init__(self, columns: str | list[str], lower_bound: int | float, upper_bound: int | float) -> None:
        """Check if a set of columns are greater than a value or another
        column.

        Args:
            columns (str | list[str] | None): List of columns on which the constraint applies to
            value (float | str): value to consider. If value is a string, it must be a column.

        Returns:
            BetweenRowConstraint: BetweenRowConstraint constraint instance
        """
    def validate(self, dataset: Dataset) -> pdDataFrame:
        """Validate the constraint against a dataset.

         Args:
            dataset (Dataset): Dataset to test

        Returns:
            pdDataFrame: constraint mask
        """

class Positive(GreaterThan):
    def __init__(self, columns: str | list[str]) -> None:
        """Check if a set of columns are greater than 0.

        Args:
            columns (str | list[str] | None): List of columns on which the constraint applies to
            value (float | str): value to consider. If value is a string, it must be a column.

        Returns:
            Positive: Positive constraint instance
        """
    def validate(self, dataset: Dataset) -> pdDataFrame:
        """Validate the constraint against a dataset.

        Args:
            dataset (Dataset): Dataset to test

        Returns:
            pdDataFrame: constraint mask
        """

class BetweenDates(RowConstraint):
    def __init__(self, constrained_column: str, reference_column: str, lower_bound: int, upper_bound: int) -> None:
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
    def validate(self, dataset: Dataset) -> pdDataFrame:
        """Validate the constraint against a dataset.

         Args:
            dataset (Dataset): Dataset to test

        Returns:
            pdDataFrame: constraint mask
        """

class IncludedIn(RowConstraint):
    def __init__(self, column: str, values: list[Any] | Any) -> None:
        """Check if the column values are included in a specified list of
        values.

        Args:
            column (str): Column on which the constraint is applied.
            values (list): List of specified values (or a single value).

        Returns:
            IncludedIn: IncludedIn constraint instance
        """
    def validate(self, dataset: Dataset) -> pdDataFrame:
        """Validate the constraint against a dataset.

         Args:
            dataset (Dataset): Dataset to test

        Returns:
            pdDataFrame: constraint mask
        """

class Regex(RowConstraint):
    def __init__(self, column: str, regex: str) -> None:
        """Check if the column values match a regular expression.

        Args:
            column (str): Column on which the constraint is applied.
            regex (str): Regular expression.

        Returns:
            Regex: Regex constraint instance
        """
    def validate(self, dataset: Dataset) -> pdDataFrame:
        """Validate the constraint against a dataset.

         Args:
            dataset (Dataset): Dataset to test

        Returns:
            pdDataFrame: constraint mask
        """

class CombineConstraints(RowConstraint):
    class Operation(Enum):
        MERGE = 'AND'
        CHAIN = 'XAND'
    def __init__(self, constraints: list[RowConstraint], operation: Operation = ...) -> None:
        """Combines a list of constraints.

        Args:
            constraints (str): List of constraints.
            operation (Operation): Operation that specifies how to combine. Defaults to CHAIN.

        Returns:
            CombinedConstraints: CombinedConstraints constraint instance
        """
    def validate(self, dataset: Dataset) -> pdDataFrame:
        """Validate the constraint against a dataset.

         Args:
            dataset (Dataset): Dataset to test

        Returns:
            pdDataFrame: constraint mask
        """
