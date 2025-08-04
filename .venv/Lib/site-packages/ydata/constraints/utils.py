from typing import Callable, List, Tuple

from pandas import DataFrame as pdDataFrame

from ydata.constraints.exceptions import NotValidatedError
from ydata.dataset.dataset import Dataset


# @typechecked
def check_row_constraint(
    dataset: Dataset, check: Callable, entity: str | None = None
) -> pdDataFrame:
    """Check a row constraint.

    It does not apply for serie constraint.

    Args:
        dataset (Dataset): Dataset on which to check the constraint
        check (Callable): Serie constraint function
        entity (str): Entity on which the constraint check should be conditioned

    Returns:
        pdDataFrame: result of the constraint
    """
    if entity:
        mask = (
            dataset._data.groupby(by=entity)
            .apply(lambda x: x.apply(check, axis=1), meta=(None, bool))
            .compute()
        )
        mask = mask.reset_index(level=0, drop=True)
        return mask
    else:
        return pdDataFrame(dataset._data.apply(check, axis=1).compute())


# @typechecked
def preprocess_series(
    dataset: Dataset, available_columns: List[str]
) -> Tuple[Dataset, int]:
    """Preprocess a dataset to check a serie constraint.

    Args:
        dataset (Dataset): Dataset to prepare
        available_columns (List[str]): List of columns to consider

    Returns:
        Dataset: prepared dataset
        int: max offset (order) in the relation
    """
    columns = []
    columns_to_offset = {}
    max_offset = 0
    for c in available_columns:
        if "|n" in c:
            if c.endswith("|n"):  # <column>|n
                columns.append(c[:-2])
                if c[:-2] not in columns_to_offset:
                    columns_to_offset[c[:-2]] = []
                columns_to_offset[c[:-2]
                                  ].append({"target_name": c, "offset": 0})
            else:
                t = c.split("|n")
                col_name = t[0]
                offset = -1 * int(t[1])
                if col_name not in columns_to_offset:
                    columns_to_offset[col_name] = []
                columns_to_offset[col_name].append(
                    {"target_name": c, "offset": offset})
                max_offset = max(max_offset, offset)
    all_cols = list(set(columns + list(columns_to_offset.keys())))
    df = dataset[all_cols]._data.compute()
    for c, C in columns_to_offset.items():
        for d in C:
            df[d["target_name"]] = df[c].shift(d["offset"])
    return Dataset(df), max_offset


# @typechecked
def check_series_relation(
    dataset: Dataset, check: Callable, available_columns: List[str]
):
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
    df, offset = preprocess_series(dataset, available_columns)
    data = df._data.compute()
    mask = check(data)
    mask.loc[:offset] = True
    return mask


# @typechecked
def require_validate(f: Callable) -> Callable:
    """Decorator to check if an instance of constraint engine has been
    validated prior calling the member."""

    def wrapper(*args, **kwargs):
        if not args[0]._validated:
            raise NotValidatedError(
                "The constraints were not validated against a dataset. Validate a dataset first!"
            )
        return f(*args, **kwargs)

    return wrapper
