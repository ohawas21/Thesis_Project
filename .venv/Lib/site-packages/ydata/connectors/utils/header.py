"""Function with the logic to validate datasets header formats."""
from re import compile, fullmatch
from typing import List, Optional

from ydata.connectors.logger import logger


def _validate_column_names(columns: List[str]) -> bool:
    """Applies a regex pattern to validate a list of column names.

    Returns True in case all names are considered valid. Returns False
    if any name is invalid.
    """
    pattern = compile(
        "".join([r"^(([-+]?(?:\d*\.\d*))|",
                 r"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}))$"])
    )  # 1. float numbers; 2. emails
    return not any([fullmatch(pattern, name) for name in columns])


def validate_header(has_header: bool, header_cols: List[str], columns: Optional[list]):
    # Validating/Creating headers
    replace = False
    if not has_header:
        header = None
        replace = True
        if columns and _validate_column_names(columns):
            assert len(columns) == len(
                header_cols
            ), "The number of column names does not match the number of columns."
            # validate whether the column names follow a certain regex
            names = columns
        else:
            # Creates a dummy header following the pattern Columns_i
            names = [f"Column_{i}" for i in range(len(header_cols))]
            logger.warning(
                "Column names are invalid and were renamed to a compatible format."
            )
    else:
        header = 0
        if _validate_column_names(header_cols):
            names = header_cols
        else:
            replace = True
            names = [f"Column_{i}" for i in range(len(header_cols))]
            logger.warning(
                "Column names are invalid and were renamed to a compatible format."
            )
    # Validate the regex of the header
    return replace, header, names
