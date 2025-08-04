import os
from enum import Enum


class FileType(Enum):
    AVRO = "avro"
    CSV = "csv"
    PARQUET = "parquet"


def infer_file_type(path: str):
    _, ext = os.path.splitext(path)

    if ext == ".csv":
        return FileType.CSV
    elif ext == ".parquet":
        return FileType.PARQUET
    elif ext == ".avro":
        return FileType.AVRO
    else:
        return None
