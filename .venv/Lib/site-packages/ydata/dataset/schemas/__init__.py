"""The :mod:`ydata.dataset.schemas` Model that gathers the dataset information
including its metadata."""
from ydata.dataset.schemas.datasets_schema import DatasetSchema, MultiTableSchema, RelationType
from ydata.dataset.schemas.rdbms_schema import Schema as RDBMSSchema
from ydata.dataset.schemas.rdbms_schema import Table

__all__ = ["Table",
           "RDBMSSchema",
           "DatasetSchema",
           "RelationType",
           "MultiTableSchema"]
