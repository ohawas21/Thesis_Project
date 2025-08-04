from pandas import DataFrame as pdDataFrame

from ydata.dataset.schemas.datasets_schema import MultiTableSchema, TableSchema
from ydata.metadata import Metadata, MultiMetadata
from ydata.utils.data_types import DataType, VariableType


def rename_column_names(table: str, table_data: pdDataFrame) -> pdDataFrame:
    table_data = table_data.rename(
        columns={col: f"{table}.{col}" for col in table_data.columns})
    return table_data


def add_string_missing_values_placeholder(
    table: str,
    table_data: pdDataFrame,
    table_metadata: Metadata,
    table_keys: list[str],
    placeholder: str = "MISSING"
):
    non_date_cols = [
        f"{table}.{name}"
        for name, column in table_metadata.columns.items()
        if column.datatype != DataType.DATE
        and f"{table}.{name}" in table_data.columns
        and column.vartype == VariableType.STR
        and name not in table_keys
    ]
    table_data[non_date_cols] = table_data[non_date_cols].fillna(placeholder)
    return table_data


def update_references_to_attribute_tables(
    table: str,
    table_data: pdDataFrame,
    metadata: MultiMetadata,
    table_schema: TableSchema,
    attribute_tables: dict,
):
    """Update metadata datatypes of foreign keys that references an attribute
    table. Obs. changes the metadata inplace;

    Args:
        table (str): table name
        table_data (pdDataFrame): table data
        metadata (Metadata): database metadata
        table_schema (TableSchema): table database schema
        attribute_tables (dict): attribute tables models
    """
    # cast fks to attribute tables to categorical
    fk_columns = {}
    for fk in table_schema.foreign_keys:
        if fk.parent_table in attribute_tables:
            fk_columns[fk.column] = "categorical"
            origin = metadata[fk.parent_table].summary["value_counts"][fk.parent_column]
            faker = attribute_tables[fk.parent_table]["model"].value_counts[fk.parent_column]
            col = f"{table}.{fk.column}"
            mapping = {s: f for s, f in zip(origin.index, faker.index)}
            table_data.loc[:, col] = table_data[col].map(mapping)

    if fk_columns:
        metadata[table].update_datatypes(fk_columns)


def get_tables_to_encode(
    schema: MultiTableSchema,
    attribute_tables: set[str]
) -> set[str]:
    """List parents of multi-parent child tables.

    Args:
        schema (MultiTableSchema): database schema
        attribute_tables (list[str]): list of attribute tables.
            Attribute tables are not considered as parent tables
    """
    parents = set()
    for table in schema.values():
        table_parents = set(
            fk.parent_table for fk in table.foreign_keys
            if fk.parent_table not in attribute_tables
        )
        if len(table_parents) > 1:
            parents |= table_parents

    return parents
