from operator import itemgetter
from random import choices

from networkx import DiGraph
from pandas import DataFrame as pdDataFrame
from pandas import Series as pdSeries
from pandas import merge as pdmerge

from ydata.dataset.schemas.datasets_schema import MultiTableSchema, TableSchema
from ydata.synthesizers.calculated_features import CalculatedFeature
from ydata.synthesizers.multitable.calculated_features import get_tables_from_columns


def _topological_sort_helper(graph, node, visited, stack):
    visited[node] = True
    for neighbor in graph[node]:
        if not visited[neighbor]:
            _topological_sort_helper(graph, neighbor, visited, stack)

    # Push current node to stack which stores result
    stack.insert(0, node)


def topological_sort(graph: dict[str, set[str]]) -> list[str]:
    """Sort graph nodes in topological order.

    Args:
        graph (dict[str, set[str]]): graph to sort.

    Returns:
        list[str]: sorted list of nodes
    """
    # Mark all the vertices as not visited
    visited = {k: False for k in graph.keys()}
    stack = []

    for neighbor in graph.keys():
        if not visited[neighbor]:
            _topological_sort_helper(graph, neighbor, visited, stack)
    return stack


def _dfs_sort_helper(graph, parent, visited: list, stack):
    for neighbor in graph[parent]:
        if neighbor not in visited:
            visited.add(neighbor)
            stack.append((neighbor, parent))
            _dfs_sort_helper(graph, neighbor, visited, stack)


def dfs_sort(graph: dict[str, set[str]], starting_nodes: list[str] | None) -> list[str]:
    """Sort graph nodes in using depth first search.

    Args:
        graph (dict[str, set[str]]): graph to sort.
        starting_nodes (list[str] | None): List with the topological order of nodes.

    Returns:
        list[str]: sorted list of nodes
    """
    visited = set()
    stack = []
    starting_nodes = starting_nodes if starting_nodes else []
    starting_nodes += [node for node in graph.keys()
                       if node not in starting_nodes]
    for node in starting_nodes:
        if node not in visited:
            visited.add(node)
            stack.append((node, ""))
            _dfs_sort_helper(graph, node, visited, stack)
    return stack


def get_multitable_synthesis_order(schema: dict):
    """Get the sinthetization order of table in the schema.

    Args:
        schema (dict): dictionary depicting the database schema.

    Returns:
        list[str]: tables synthetization order
    """
    mt_schema = MultiTableSchema(schema)
    relationships = get_table_relationships(mt_schema)
    digraph = {k: set() for k in schema}
    for child, parent, _ in relationships:
        digraph[parent].add(child)
    order = topological_sort(digraph)
    return order


def drop_unmatching_keys(
    table_schema: TableSchema,
    sample_data: pdDataFrame,
    sample_tables: dict[str, pdDataFrame],
) -> pdDataFrame:
    """Drop rows with invalid FK's.

    Args:
        table_schema (TableSchema): schema describing the table relationships.
        sample_data (pdDataFrame): synthetic data.
        sample_tables (dict): collection of synthesized tables.

    Returns:
        pdDataFrame: synthetic data with invalid keys filtered out
    """
    for fk in table_schema.foreign_keys:
        if fk.parent_table in sample_tables:
            parent_table = sample_tables[fk.parent_table]
            if "." in parent_table.columns[0]:
                valid_keys = sample_tables[fk.parent_table][
                    f"{fk.parent_table}.{fk.parent_column}"].unique()
            else:
                valid_keys = sample_tables[fk.parent_table][fk.parent_column].unique(
                )
            mask = sample_data[fk.column].isin(valid_keys)

            sample_data = sample_data.loc[mask].reset_index(
                drop=True)
    return sample_data


def replace_for_valid_keys(
    table_schema: TableSchema,
    sample_data: pdDataFrame,
    sample_tables: dict[str, pdDataFrame],
    missing_values: dict[str, int],
    reference_values: dict[str, pdDataFrame],
) -> pdDataFrame:
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
    # fixme only work for 1-N
    for fk in table_schema.foreign_keys:
        if fk.parent_table in sample_tables:
            parent_table = sample_tables[fk.parent_table]
            if parent_table.columns[0].startswith(f"{parent_table}."):
                valid_keys = sample_tables[fk.parent_table][f"{fk.parent_table}.{fk.parent_column}"].unique(
                )
                parent_dtype = sample_tables[fk.parent_table][
                    f"{fk.parent_table}.{fk.parent_column}"].dtype
            else:
                valid_keys = sample_tables[fk.parent_table][fk.parent_column].unique(
                )
                parent_dtype = sample_tables[fk.parent_table][fk.parent_column].dtype
        else:
            valid_keys = reference_values[fk.parent_table][fk.parent_column]
            parent_dtype = reference_values[fk.parent_table][fk.parent_column].dtype

        has_missing = missing_values.get(fk.column, 0) > 0
        mask = ~sample_data[fk.column].isin(valid_keys)
        if has_missing:
            mask = mask & ~sample_data[fk.column].isna()

        if mask.any():
            new_valid_keys = choices(
                valid_keys, k=mask.sum())
            dtype = sample_data[fk.column].dtype if parent_dtype not in [
                'object', 'string'] else 'string'

            sample_data.loc[mask, fk.column] = pdSeries(
                new_valid_keys, dtype=dtype).to_list()
    return sample_data


def is_parent_persisted(table_schema: TableSchema, persisted: set) -> bool:
    """Check if all parent tables were already persisted.

    Args:
        table_schema (TableSchema): table schema.
        persisted (set): collection of persisted tables.

    Returns:
        bool: True if all parent tables were persisted, False otherwise.
    """
    return all([fk.parent_table in persisted for fk in table_schema.foreign_keys])


def get_tables_to_persist(
    tables: list,
    sampled_tables: dict,
    persisted_tables: set,
    schema: MultiTableSchema,
    composite_keys: dict,
    calculated_features: list[CalculatedFeature]
) -> set[str]:
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
    tables_to_sample = {
        t for t in tables
        if t not in sampled_tables and t not in persisted_tables
    }

    necessary_tables = set()
    for t in tables_to_sample:
        # don't persit parent tables until all children are generated
        necessary_tables |= {fk.parent_table for fk in schema[t].foreign_keys}

    generated_tables = set(sampled_tables.keys()) | persisted_tables
    for parent_table in sampled_tables:
        # don't persit tables that have composite_keys dependencies
        if parent_table in composite_keys:
            keys = composite_keys[parent_table]
            # Does the table has a ungenerated child?
            if any(
                ck["table"] not in generated_tables
                for ck in keys
            ):
                necessary_tables.add(parent_table)

    for cf in calculated_features:
        feature_table = get_tables_from_columns(cf.features).pop()
        source_table = get_tables_from_columns(cf.calculated_from).pop()
        if feature_table in sampled_tables and source_table in tables_to_sample:
            necessary_tables.add(feature_table)
        elif feature_table in tables_to_sample and source_table in sampled_tables:
            necessary_tables.add(feature_table)

    necessary_tables |= tables_to_sample

    tables_to_persist = set()
    for table in sampled_tables:
        if (
            table not in necessary_tables and
            is_parent_persisted(
                schema[table],
                persisted_tables
            )
        ):
            tables_to_persist.add(table)
    return tables_to_persist


def get_table_relationships(relations: MultiTableSchema):
    fks = []

    for relation in relations.values():
        parent_table_references = {}
        for fk in relation.foreign_keys:
            # ignores self reference
            if fk.table == fk.parent_table:
                continue
            if fk.parent_table not in parent_table_references:
                parent_table_references[fk.parent_table] = []
            parent_table_references[fk.parent_table].append(
                (
                    fk.table,
                    fk.parent_table,
                    {
                        "left_on": f"{fk.parent_table}.{fk.parent_column}",
                        "right_on": f"{fk.table}.{fk.column}",
                    },
                )
            )
        for reference in parent_table_references.values():
            if len(reference) == 1:
                fks.extend(reference)
            else:  # Handles composite foreign keys
                table = reference[0][0]
                parent_table = reference[0][1]
                left_on = []
                right_on = []
                for ref in reference:
                    left = ref[-1]["left_on"]
                    right = ref[-1]["right_on"]
                    # ignores multiple references for the same column, and just consider the first
                    if left not in left_on and right not in right_on:
                        left_on.append(left)
                        right_on.append(right)
                left_on = left_on if len(left_on) > 1 else left_on[0]
                right_on = right_on if len(right_on) > 1 else right_on[0]
                fks.append(
                    (
                        table,
                        parent_table,
                        {
                            "left_on": left_on,
                            "right_on": right_on,
                        },
                    )
                )

    return fks


def get_relationship(relationships, left, right):
    for rel in relationships:
        if (
            (left == rel[0] and right == rel[1])
            or (left == rel[1] and right == rel[0])
        ):
            return rel[2]
    return None


def merge_tables(relationship, tables_df, reference_table, columns: dict | None = None):
    if isinstance(relationship["left_on"], str):
        left = relationship["left_on"].split('.', 1)[0]
    else:
        left = relationship["left_on"][0].split('.', 1)[0]

    if isinstance(relationship["right_on"], str):
        right = relationship["right_on"].split('.', 1)[0]
    else:
        right = relationship["right_on"][0].split('.', 1)[0]

    how = "left" if left == reference_table else "right"

    left_table = tables_df[left]
    right_table = tables_df[right]
    if columns:
        if left in columns:
            left_table = left_table[columns[left]]
        if right in columns:
            right_table = right_table[columns[right]]
    return pdmerge(
        left_table,
        right_table,
        how=how,
        left_on=relationship["left_on"],
        right_on=relationship["right_on"],
    )


def build_relation_graph(relations: dict):
    """Create a relationship graph based on the table schemas.

    Args:
        relations (dict[str, TableSchema]):  collection of tables' schemas.

    Returns:
        a tuple containing:
            graph (DiGraph): a directional graph with the relationships
            relationships: a tuple with he names of the child and parent tables, and dict with columns information for joining.
    """
    graph = DiGraph()
    graph.add_nodes_from(relations.keys())
    relationships = get_table_relationships(relations)
    graph.add_edges_from(relationships)

    return graph, relationships


def get_merge_sequence(graph) -> list:
    """Get tables merge order.

    Args:
        graph (DiGraph): relationships graph

    Returns:
        list: sequence of tables pairs to merge
    """
    visitors = []
    while len(list(graph.nodes)) > 0:
        right_node = sorted(
            list(graph.in_degree(graph.nodes)),
            key=itemgetter(1)
        )[0]

        for left_node in list(graph.successors(right_node[0])):
            data = graph.edges[right_node[0], left_node]
            data.update({"left": left_node, "right": right_node[0]})
            visitors.append(data)
        graph.remove_node(right_node[0])

    return visitors


def get_expected_size(nrows: int, fraction: float):
    """Get expected size of a table.

    Args:
        nrows (int): number of rows on the original table
        fraction (float): expected dataset fraction

    Returns:
        int: expected number of rows
    """
    if nrows == 0:
        return 0
    expected = int(nrows * fraction)
    return expected if expected > 0 else 1
