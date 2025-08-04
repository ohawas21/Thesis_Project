from pandas import DataFrame as pdDataFrame
from typing import Callable
from ydata.dataset.dataset import Dataset

def check_row_constraint(dataset: Dataset, check: Callable, entity: str | None = None) -> pdDataFrame:
    """Check a row constraint.

    It does not apply for serie constraint.

    Args:
        dataset (Dataset): Dataset on which to check the constraint
        check (Callable): Serie constraint function
        entity (str): Entity on which the constraint check should be conditioned

    Returns:
        pdDataFrame: result of the constraint
    """
def preprocess_series(dataset: Dataset, available_columns: list[str]) -> tuple[Dataset, int]:
    """Preprocess a dataset to check a serie constraint.

    Args:
        dataset (Dataset): Dataset to prepare
        available_columns (List[str]): List of columns to consider

    Returns:
        Dataset: prepared dataset
        int: max offset (order) in the relation
    """
def check_series_relation(dataset: Dataset, check: Callable, available_columns: list[str]):
    """Wrapper to check the series constraint.

    The idea is to analyse the required columns to understand the order of the relation.
    Then, we create a shifted version of the columns according to the order in the relation.

    Remark: We assume that the relation for the initial values are true

    Args:
        dataset (Dataset): Dataset on which to check the constraint
        check (Callable): Serie constraint function
        available_columns (List[str]): List of columns used by the function `check`


    Returns:
        ConstraintEngine: ConstraintEngine instance
    """
def require_validate(f: Callable) -> Callable:
    """Decorator to check if an instance of constraint engine has been
    validated prior calling the member."""
