from typing import List, Optional

from ydata.dataset import Dataset
from ydata.utils.exceptions import ColumnNotFoundError


def validate_columns_in_dataset(
    dataset: Dataset, columns: Optional[List[str]] = None
) -> None:
    if columns is not None:
        invalid_cols = [c for c in columns if c not in dataset.columns]
        if len(invalid_cols):
            raise ColumnNotFoundError(
                "The following columns are not in the dataset: {}".format(
                    ", ".join(invalid_cols)
                )
            )
