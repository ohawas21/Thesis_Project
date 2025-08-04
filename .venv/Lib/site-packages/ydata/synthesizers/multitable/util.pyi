from pandas import DataFrame as pdDataFrame
from ydata.dataset.schemas.datasets_schema import MultiTableSchema, TableSchema as TableSchema
from ydata.synthesizers.calculated_features import CalculatedFeature

def topological_sort(graph: dict[str, set[str]]) -> list[str]:
    """Sort graph nodes in topological order.

    Args:
        graph (dict[str, set[str]]): graph to sort.

    Returns:
        list[str]: sorted list of nodes
    """
def dfs_sort(graph: dict[str, set[str]], starting_nodes: list[str] | None) -> list[str]:
    """Sort graph nodes in using depth first search.

    Args:
        graph (dict[str, set[str]]): graph to sort.
        starting_nodes (list[str] | None): List with the topological order of nodes.

    Returns:
        list[str]: sorted list of nodes
    """
def get_multitable_synthesis_order(schema: dict):
    """Get the sinthetization order of table in the schema.

    Args:
        schema (dict): dictionary depicting the database schema.

    Returns:
        list[str]: tables synthetization order
    """
def drop_unmatching_keys(table_schema: TableSchema, sample_data: pdDataFrame, sample_tables: dict[str, pdDataFrame]) -> pdDataFrame:
    """Drop rows with invalid FK's.

    Args:
        table_schema (TableSchema): schema describing the table relationships.
        sample_data (pdDataFrame): synthetic data.
        sample_tables (dict): collection of synthesized tables.

    Returns:
        pdDataFrame: synthetic data with invalid keys filtered out
    """
def replace_for_valid_keys(table_schema: TableSchema, sample_data: pdDataFrame, sample_tables: dict[str, pdDataFrame], missing_values: dict[str, int], reference_values: dict[str, pdDataFrame]) -> pdDataFrame:
    """Replace invalid FK's for valid ones at random.

    Args:
        table_schema (TableSchema): schema describing the table relationships.
        sample_data (pdDataFrame): synthetic data.
        sample_tables (dict): collection of synthesized tables.
        missing_values (dict): indicates the presence of missing values on the original tables.
        reference_values (dict): valid key values for persisted tables.

    Returns:
        pdDataFrame: synthetic data with invalid FK's replaced
    """
def is_parent_persisted(table_schema: TableSchema, persisted: set) -> bool:
    """Check if all parent tables were already persisted.

    Args:
        table_schema (TableSchema): table schema.
        persisted (set): collection of persisted tables.

    Returns:
        bool: True if all parent tables were persisted, False otherwise.
    """
def get_tables_to_persist(tables: list, sampled_tables: dict, persisted_tables: set, schema: MultiTableSchema, composite_keys: dict, calculated_features: list[CalculatedFeature]) -> set[str]:
    """Get the list of tables that can be safely flushed from memory.

    Args:
        tables (list): list of table names.
        sampled_tables (dict): collection of tables already synthesized.
        persisted_tables (set): collection of tables already persisted.
        schema (MultiTableSchema): database schema.
        composite_keys (dict): manual composite keys dependencies.
        calculated_features (list[CalculatedFeature]): list of calculated features.

    Returns:
        set[str]: colletion of tables that can be safely removed from memory.
    """
def get_table_relationships(relations: MultiTableSchema): ...
def get_relationship(relationships, left, right): ...
def merge_tables(relationship, tables_df, reference_table, columns: dict | None = None): ...
def build_relation_graph(relations: dict):
    """Create a relationship graph based on the table schemas.

    Args:
        relations (dict[str, TableSchema]):  collection of tables' schemas.

    Returns:
        a tuple containing:
            graph (DiGraph): a directional graph with the relationships
            relationships: a tuple with he names of the child and parent tables, and dict with columns information for joining.
    """
def get_merge_sequence(graph) -> list:
    """Get tables merge order.

    Args:
        graph (DiGraph): relationships graph

    Returns:
        list: sequence of tables pairs to merge
    """
def get_expected_size(nrows: int, fraction: float):
    """Get expected size of a table.

    Args:
        nrows (int): number of rows on the original table
        fraction (float): expected dataset fraction

    Returns:
        int: expected number of rows
    """
