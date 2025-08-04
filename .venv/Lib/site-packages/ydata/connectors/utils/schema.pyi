from pandas.io.parsers import TextFileReader as TextFileReader
from ydata.dataset.filetype import FileType

def calculate_schema(chunks: TextFileReader) -> dict:
    """Calculated the schema based on a pandas TextFileReader(chunks) chunks
    (TextFileReader): Iterator with the chunks definition."""
def get_schema(file_path: list | str, file_type: FileType, sep: str = ';', storage_options: dict | None = None) -> dict:
    """Calculates the best fit schema for the dataset to be consumed by DASK.

    It returns a function with the columns and respective variables
    types
    """
