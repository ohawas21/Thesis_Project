from ydata.dataset.multidataset import MultiTableSchema

class SchemaValidator:
    def check_independent_tables(self, schema: MultiTableSchema):
        """Identify independent tables in the schema.

        An table is considered independent if it don't have
        relationships with any other table within the schema.
        """
    def check_no_relations(self, schema: MultiTableSchema):
        """Identify if all tables in the schema are independent."""
    def check_circular_relationships(self, schema: MultiTableSchema):
        """Identify potential circular relationship within the schema."""
    def check_indirect_relationships(self, schema: MultiTableSchema):
        """Identify tables that share references to the same table/column
        without an explicit relation between them."""
    def check_self_reference(self, schema: MultiTableSchema):
        """Identify tables that reference itself."""
    def get_warnings(self, schema: MultiTableSchema): ...
