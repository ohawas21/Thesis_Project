from enum import Enum

class FileType(Enum):
    AVRO = 'avro'
    CSV = 'csv'
    PARQUET = 'parquet'

def infer_file_type(path: str): ...
