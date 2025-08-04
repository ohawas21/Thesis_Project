"""Python file with the Dataset  and multidataset schemas dataclass."""
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Union

from pydantic.v1 import BaseModel, Extra, Field

from ydata.utils.data_types import VariableType


@dataclass(frozen=True)
class DatasetSchema:
    column: str
    vartype: VariableType
    format: str | None = None


class RelationType(Enum):
    ONE_TO_MANY = '1-n'
    ONE_TO_ONE = '1-1'
    MANY_TO_MANY = 'n-n'


class ForeignReference(BaseModel):
    table: str
    column: str
    parent_table: str
    parent_column: str
    relation_type: RelationType = RelationType.MANY_TO_MANY

    def __str__(self) -> str:
        return f"{self.table}.{self.column} ref to {self.parent_table}.{self.parent_column}"

    def __hash__(self):
        return hash(str(self))


class TableSchema(BaseModel):
    _schema: dict = Field(default_factory=list, alias='schema')
    primary_keys: list[str] = Field(default_factory=list)
    foreign_keys: list[ForeignReference] = Field(
        default_factory=list[ForeignReference])
    columns: dict = Field(default_factory=dict, alias='columns')

    class Config:
        extra = Extra.ignore

    def get_keys(self):
        keys = self.primary_keys + [fk.column for fk in self.foreign_keys]
        return keys


class MultiTableSchema(dict[str, TableSchema]):
    def __init__(self, data: Union[dict, "RDBMS_Schema"] | None, tables: list[str] | None = None):  # noqa: F821
        from ydata.dataset.schemas.rdbms_schema import Schema as RDBMS_Schema
        if data is None:
            data = {}

        if isinstance(data, RDBMS_Schema):
            data = self.__from_RDBMS_schema(data)

        # If the input data does not have the table reference for FK, let's add it
        for t, v in data.items():
            if isinstance(v, dict):
                for fk in v.get('foreign_keys', []):
                    if isinstance(fk, dict) and 'table' not in fk:
                        fk['table'] = t

        for t, v in data.items():
            if tables is None or t in tables:
                if isinstance(v, TableSchema):
                    self[t] = v
                else:
                    self[t] = TableSchema(**v)

        self.__update_foreign_key_relations()

        self.composite_keys = {}

    def add_composite_keys(self, table: str, columns: list[str], parent_table: str, parent_columns: list[str]):
        """Define a composite key for a table. The keys are expected to be in
        the same order.

        Obs.: if a composite key already exists for the table, replaces for one with the new paramentes.

        Args:
            table (str): table name.
            columns (list[str]): columns that composes the key.
            parent_table (str): parent table name
            parent_columns (list[str]): columns that compose the parent table key,
        """

        self.composite_keys[table] = {
            "columns": columns,
            "parent_table": parent_table,
            "parent_columns": parent_columns,
        }

    @property
    def tables(self):
        return list(self.keys())

    def dict(self) -> dict:
        return {k: v.dict() for k, v in self.items()}

    @property
    def foreign_keys(self) -> list[ForeignReference]:
        return [fk for t in self.values() for fk in t.foreign_keys]

    def add_foreign_key(self, table: str, column: str, parent_table: str, parent_column: str, relation_type: str | RelationType = RelationType.MANY_TO_MANY):
        fk = ForeignReference(table=table, column=column,
                              parent_table=parent_table, parent_column=parent_column, relation_type=RelationType(relation_type))
        if fk not in self[table].foreign_keys:
            self[table].foreign_keys.append(fk)
        self.__update_foreign_key_relations()

    def add_primary_key(self, table: str, column: str):
        if column not in self[table].primary_keys:
            self[table].primary_keys.append(column)
        self.__update_foreign_key_relations()

    def filter(self, tables: Union[list, str]):
        """
        Method that return a new Schema containing only the information for the selected tables
        Parameters
        ----------
        tables a table or a list of tables
        Returns A MultiTable schema
        -------
        """
        if isinstance(tables, str):
            tables = [tables]

        return MultiTableSchema(data=self.dict(),
                                tables=tables)

    def __update_foreign_key_relations(self):
        for table in self.values():
            table.foreign_keys = [
                fk for fk in table.foreign_keys if fk.parent_table in self]
            for fk in table.foreign_keys:
                # Foreign keys are by definition unique, so update to 1-n relation
                if fk.relation_type == RelationType.MANY_TO_MANY and fk.parent_column in self[fk.parent_table].primary_keys:
                    fk.relation_type = RelationType.ONE_TO_MANY

    @staticmethod
    def __from_RDBMS_schema(data) -> dict:
        data = asdict(data)['tables']
        schema_json = {}
        for table, content in data.items():
            schema_json[table] = {
                'primary_keys': [],
                'foreign_keys': [],
                'columns': {}
            }
            for col in content['columns']:
                if col['primary_key']:
                    schema_json[table]['primary_keys'].append(col['name'])
                for fk in col['foreign_keys']:
                    parent_table, parent_column = fk['parent'].split('.')
                    schema_json[table]['foreign_keys'].append({
                        'column': col['name'],
                        'parent_table': parent_table,
                        'parent_column': parent_column
                    })
                schema_json[table]['columns'][col['name']
                                              ] = col['variable_type']
        return schema_json
