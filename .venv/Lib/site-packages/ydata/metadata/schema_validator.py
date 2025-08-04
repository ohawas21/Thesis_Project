from collections import defaultdict

from ydata.dataset.multidataset import MultiTableSchema
from ydata.metadata.multitable_warning import (CircularReferenceWarning, IndependentTableWarning,
                                               IndirectRelationshipsWarning, MultiTableWarningType,
                                               NoRelationshipsWarning, SelfReferenceWarning)


class SchemaValidator:
    def check_independent_tables(self, schema: MultiTableSchema):
        """Identify independent tables in the schema.

        An table is considered independent if it don't have
        relationships with any other table within the schema.
        """
        related_tables = set()
        for table in schema.values():
            for fk in table.foreign_keys:
                related_tables.add(fk.table)
                related_tables.add(fk.parent_table)
        independent_tables = set(schema.keys()) - related_tables

        if independent_tables:
            return IndependentTableWarning([list(independent_tables)])
        return None

    def check_no_relations(self, schema: MultiTableSchema):
        """Identify if all tables in the schema are independent."""
        if len(schema.foreign_keys) == 0:
            return NoRelationshipsWarning([list(schema.keys())])
        return None

    def _find_paths(self, schema: MultiTableSchema):
        """Finds cicles within a database schema."""
        # Helper function to perform DFS and find paths
        def dfs(table: str, visited: set[str], path: list[str]):
            # If table table is visited we found a circle
            parents = {fk.parent_table for fk in schema[table].foreign_keys}
            if len(parents) == 0 and len(path) > 1:
                discovered_paths.append(path)
                return
            if table in visited:
                discovered_paths.append(path)
                return

            visited.add(table)
            for parent in parents:
                dfs(parent, visited.copy(), path + [parent])

        discovered_paths = []
        for table in schema.keys():
            dfs(table, set(), [table])

        # Remove duplicates
        unique_paths = []
        paths = set()
        for path in discovered_paths:
            str_path = ''.join(path)
            if str_path not in paths:
                unique_paths.append(path)
                paths.add(str_path)

        return unique_paths

    def check_circular_relationships(self, schema: MultiTableSchema):
        """Identify potential circular relationship within the schema."""
        def is_circular_paths(path_a: list[str], path_b: list[str]) -> bool:
            """Check if the paths share the same start and end tables.

            Args:
                path_a (list[str]): relationship path between tables.
                path_b (list[str]): relationship path between tables.
            """
            return (
                path_a['values'][0] == path_b['values'][0]
                and path_a['values'][-1] == path_b['values'][-1]
            )

        database_paths = self._find_paths(schema)
        database_paths = [
            {'values': path, 'circular': False}
            for path in database_paths
        ]

        for i, path in enumerate(database_paths):
            # Validates wether the path starts and ends with the same table
            if path['values'][0] == path['values'][-1]:
                path['circular'] = True

            elif i < len(database_paths) - 1:
                # Validates wether there are more than 1 path that starts and ends with the same tables, where start_table != end_table
                for j in range(i + 1, len(database_paths)):
                    if is_circular_paths(path, database_paths[j]):
                        path['circular'] = True
                        database_paths[j]['circular'] = True

        circular_paths = [
            path['values']
            for path in database_paths if path['circular']
        ]
        if circular_paths:
            return CircularReferenceWarning(circular_paths)
        else:
            return None

    def check_indirect_relationships(self, schema: MultiTableSchema):
        """Identify tables that share references to the same table/column
        without an explicit relation between them."""
        references = defaultdict(set)
        for fk in schema.foreign_keys:
            references[fk.table].add(fk.parent_table)

        tables = list(references.keys())
        indirect_references = {}
        for i, table in enumerate(tables):
            for other in tables[i+1:]:
                intersection = references[table] & references[other]
                if len(intersection) > 1:
                    indirect_references[f"{table}|{other}"] = intersection

        if indirect_references:
            indirect_references = [
                ref.split("|", 1)
                for ref in indirect_references.keys()
            ]
            return IndirectRelationshipsWarning(indirect_references)
        else:
            return None

    def check_self_reference(self, schema: MultiTableSchema):
        """Identify tables that reference itself."""
        self_references = {
            fk.table for fk in schema.foreign_keys
            if fk.table == fk.parent_table
        }

        if self_references:
            return SelfReferenceWarning([list(self_references)])
        else:
            return None

    def get_warnings(self, schema: MultiTableSchema):
        warnings = {
            MultiTableWarningType.NO_RELATIONSHIPS.value: self.check_no_relations(schema),
            MultiTableWarningType.INDEPENDENT_TABLE.value: self.check_independent_tables(schema),
            MultiTableWarningType.CIRCULAR_REFERENCE.value: self.check_circular_relationships(schema),
            MultiTableWarningType.SELF_REFERENCE.value: self.check_self_reference(schema),
            MultiTableWarningType.INDIRECT_RELATIONSHIP.value: self.check_indirect_relationships(schema),
        }
        return {k: v for k, v in warnings.items() if v is not None}
