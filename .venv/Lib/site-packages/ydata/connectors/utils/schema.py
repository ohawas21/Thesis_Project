"""Get schema from CSV files related functions."""
from typing import Optional, Union

from pandas import DataFrame
from pandas.io.parsers import TextFileReader

from ydata.dataset.filetype import FileType
from ydata.connectors.readers import PandasReader


# @typechecked
def calculate_schema(chunks: TextFileReader) -> dict:
    """Calculated the schema based on a pandas TextFileReader(chunks) chunks
    (TextFileReader): Iterator with the chunks definition."""
    dtypes = []
    for i, chunk in enumerate(chunks):
        if i <= 15:
            dtypes.append(chunk.dtypes)
        else:
            break
    dtypes = DataFrame(dtypes)
    schema = dtypes.max().to_dict()
    return schema


# @typechecked
def get_schema(
    file_path: Union[list, str],
    file_type: FileType,
    sep: str = ";",
    storage_options: Optional[dict] = None,
) -> dict:
    """Calculates the best fit schema for the dataset to be consumed by DASK.

    It returns a function with the columns and respective variables
    types
    """
    read_parameters = {
        "sep": sep,
        "storage_options": storage_options,
        "chunksize": 50000,
        "low_memory": False,
    }
    if file_type == FileType.CSV:
        read_parameters["encoding_errors"] = "ignore"

    chunks = PandasReader(file_type).read(file_path, **read_parameters)
    return calculate_schema(chunks)
