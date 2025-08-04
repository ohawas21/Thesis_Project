from pandas import DataFrame as pdDataFrame
from ydata.dataset.schemas.datasets_schema import MultiTableSchema as MultiTableSchema, TableSchema as TableSchema
from ydata.metadata import Metadata as Metadata, MultiMetadata as MultiMetadata

def rename_column_names(table: str, table_data: pdDataFrame) -> pdDataFrame: ...
def add_string_missing_values_placeholder(table: str, table_data: pdDataFrame, table_metadata: Metadata, table_keys: list[str], placeholder: str = 'MISSING'): ...
def update_references_to_attribute_tables(table: str, table_data: pdDataFrame, metadata: MultiMetadata, table_schema: TableSchema, attribute_tables: dict):
    """Update metadata datatypes of foreign keys that references an attribute
    table. Obs. changes the metadata inplace;

    Args:
        table (str): table name
        table_data (pdDataFrame): table data
        metadata (Metadata): database metadata
        table_schema (TableSchema): table database schema
        attribute_tables (dict): attribute tables models
    """
def get_tables_to_encode(schema: MultiTableSchema, attribute_tables: set[str]) -> set[str]:
    """List parents of multi-parent child tables.

    Args:
        schema (MultiTableSchema): database schema
        attribute_tables (list[str]): list of attribute tables.
            Attribute tables are not considered as parent tables
    """
