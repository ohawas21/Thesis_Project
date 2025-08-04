from __future__ import annotations

from os import getenv
from pickle import HIGHEST_PROTOCOL, dump, load

from dask import compute
from numpy import average as np_average
from pandas import DataFrame
from pandas.api.types import is_numeric_dtype

from ydata.characteristics import ColumnCharacteristic
from ydata.dataset import Dataset, DatasetType, MultiDataset
from ydata.dataset.schemas.datasets_schema import ForeignReference, MultiTableSchema, RelationType, TableSchema
from ydata.metadata import Metadata
from ydata.metadata.schema_validator import SchemaValidator
from ydata.metadata.utils import dropna, value_counts
from ydata.utils.configuration import TextStyle
from ydata.utils.data_types import DataType
from ydata.utils.exceptions import InvalidDatasetTypeError
from ydata.utils.logger import utilslogger_config

logger = utilslogger_config(verbose=getenv(
    "VERBOSE", "false").lower() == "true")


def _format_count(count, total):
    ratio = count / total if total > 0 else 0
    return f"{count} ({round(100. * ratio, 2):.2f}%)"


class MultiMetadata:
    def __init__(
        self,
        multiset: MultiDataset,
        tables_metadata: dict[str, Metadata] | None = None,
        dataset_attrs: dict[str, dict] | None = None,
        dataset_type: dict[str, DatasetType | str] | None = None,
        schema: MultiTableSchema | None = None
    ) -> None:
        self._dataset_attrs = dataset_attrs
        self.dataset_type = {}
        # Metadata are computed only if not provided by the user
        tables_metadata = tables_metadata if tables_metadata is not None else {}
        self._metas = {
            k: tables_metadata[k] for k, _ in multiset.items() if k in tables_metadata
        }
        if len(multiset.keys()) > len(self._metas):
            uncomputed_metadata = set(
                multiset.keys()) - set(self._metas.keys())
            logger.info(
                f"[MULTIMETADATA] - Calculating metadata for tables: {uncomputed_metadata}.")
        self._metas.update({k: Metadata(v) if v is not None else Metadata(multiset[k])
                           for k, v in multiset.items() if k not in self._metas})

        logger.info("[MULTIMETADATA] - Initializing characteristics.")
        self.schema = schema if schema is not None else multiset.schema
        for k, m in self._metas.items():
            if m is not None:
                self._metas[k] = self._add_id_characteristics(
                    k, m, multiset[k])

        logger.info("[MULTIMETADATA] - Validating schema.")
        validator = SchemaValidator()
        self.__warnings = validator.get_warnings(self.schema)

        logger.info("[MULTIMETADATA] - Update relationship types.")
        self._update_foreign_key_relations()
        ''' Deactivate for now as it is computationally expensive
        no_elements = self._compute_zero_cardinality_fks(dataset=multiset)
        self.zero_children_relations = no_elements[0]
        self.zero_parents_relations = no_elements[1]
        '''

        dataset_type = dataset_type if dataset_type is not None else {}
        try:
            for k in self._metas.keys():
                if k in dataset_type.keys():
                    self.dataset_type[k] = DatasetType(dataset_type[k])
        except ValueError:
            raise InvalidDatasetTypeError(
                f"Provided dataset_type {dataset_type} is not valid."
            )

        self.__valid_inputs(self._dataset_attrs)
        multiset.add_observer_for_new_tables(self._new_table_callback)
        self._deferred_request_endpoint = multiset._deferred_request_endpoint()

    @property
    def warnings(self) -> dict:
        return self.__warnings

    def _new_table_callback(self, table_name: str, dataset: Dataset):
        self._metas[table_name] = Metadata(dataset)
        self._metas[table_name] = self._add_id_characteristics(
            table_name, self._metas[table_name], dataset)
        self._update_foreign_key_relations()

    def _add_id_characteristics(self, table_name: str, meta: Metadata, dataset: Dataset) -> Metadata:
        """Adds the ID characteristic to primary keys and foreign keys.

        If the metadata is lazy-computed, only the tables already
        fetched are considered.
        """
        table = self.schema.get(table_name)
        if table is not None:
            keys = []
            if table.primary_keys:
                meta.add_characteristics(
                    {pk: ColumnCharacteristic.ID for pk in table.primary_keys})
                keys.extend(table.primary_keys)

            if table.foreign_keys:
                meta.add_characteristics(
                    {fk.column: ColumnCharacteristic.ID for fk in table.foreign_keys})
                keys.extend([fk.column for fk in table.foreign_keys])

            key_types = {key: DataType.CATEGORICAL for key in keys}
            meta.update_datatypes(key_types, dataset)
        return meta

    def _update_foreign_key_relations(self):
        """Updates the type of relationships between tables based on the
        cardinality of the foreign keys.

        If the metadata is lazy-computed, only the tables already
        fetched are considered.
        """
        # TODO: Detect the relation type automatically based on the cardinality for each columns
        for k, v in self._metas.items():
            if v is not None:
                table = self.schema.get(k)
                if table is not None and table.foreign_keys:
                    for fk in table.foreign_keys:
                        if self._metas[fk.parent_table] is not None and self._metas[fk.table] is not None:
                            # Foreign key is unique in the parent table
                            if self._metas[fk.parent_table].summary['cardinality'][fk.parent_column] == self._metas[fk.parent_table].summary['nrows']:
                                if self._metas[fk.table].summary['nrows'] == self._metas[fk.parent_table].summary['cardinality'][fk.parent_column]:
                                    fk.relation_type = RelationType.ONE_TO_ONE
                                else:
                                    fk.relation_type = RelationType.ONE_TO_MANY

    def _compute_zero_cardinality_fks(self, dataset: MultiDataset):
        """ For foreign keys with 1-N cardinality, this method compute the number of rows
        in the parent table without any children, i.e. when N = 0.
        """
        tasks_children = {
            t: {}
            for t, m in self._metas.items()
            if m is not None
        }
        tasks_parents = {
            t: {}
            for t, m in self._metas.items()
            if m is not None
        }
        for table_name, m in self._metas.items():
            if m is not None:
                table = self.schema.get(table_name)
                if table is not None and table.foreign_keys:
                    child_table = dataset[table_name].to_dask()
                    if child_table is not None:
                        for fk in table.foreign_keys:
                            if fk.relation_type == RelationType.ONE_TO_MANY:
                                parent_table = dataset[fk.parent_table].to_dask(
                                )
                                if parent_table is not None:
                                    # 1-N with N = 0 (no children)
                                    count_without_children = self._metas[fk.parent_table].summary['nrows'] - parent_table[fk.parent_column].astype(
                                        'category').cat.set_categories(child_table[fk.column].unique().dropna().values).dropna().shape[0]
                                    tasks_parents[table_name][fk] = count_without_children
                                    count_without_parents = self._metas[fk.table].summary['missings'].get(fk.column, 0) + self._metas[fk.table].summary['nrows'] - child_table[fk.column].astype(
                                        'category').cat.set_categories(parent_table[fk.parent_column].unique().dropna().values).dropna().shape[0]
                                    tasks_children[table_name][fk] = count_without_parents
        return compute([tasks_parents, tasks_children])[0]

    def compute(self):
        """Request all the tables that are not available yet."""
        for key in self._metas.keys():
            if self._metas[key] is None:
                self.__getitem__(key)
        return self

    def __valid_inputs(self, dataset_attrs):
        for k in self._metas.keys():
            if k in self.dataset_type.keys():
                if self.dataset_type[k] == DatasetType.TIMESERIES:
                    assert dataset_attrs, "Timeseries dataset requires dataset_attrs."
                    assert (
                        "sortbykey" in dataset_attrs[k]
                    ), "Timeseries dataset requires sortbykey attribute."

    def __getitem__(self, key: str) -> Metadata:
        if self._metas[key] is None and self._deferred_request_endpoint is not None:
            # If the Metadata is not available, the table is requested from the MultiDataset,
            #   which will then trigger the Metadata computation through the callback function.
            self._deferred_request_endpoint.request_table(key)
        return self._metas[key]

    def items(self):
        self.compute()
        return self._metas.items()

    def keys(self):
        return self._metas.keys()

    def values(self):
        self.compute()
        return self._metas.values()

    def __iter__(self):
        self.compute()
        return self._metas.__iter__()

    def __delitem__(self, table: str):
        del self._metas[table]

    def __str__(self):
        def _get_pk_summary(pks: list[str], table: str):
            summary = []
            for pk in pks:
                for c in self[table].columns[pk].characteristics:
                    summary.append(c.value)
            return summary

        def _get_fk_summary(fk: list, table: str):
            summary = {}
            for a in fk:
                characteristics = []
                for c in self[table].columns[a].characteristics:
                    characteristics.append(c.value)
                summary[a] = characteristics
            return summary

        n_tables = len(self.schema)

        str_repr = TextStyle.BOLD + "MultiMetadata Summary \n \n" + TextStyle.END

        str_repr += (
            TextStyle.BOLD
            + "Tables Summary "
            + TextStyle.END
            + "\n"
        )
        str_repr += (
            TextStyle.BOLD
            + "Number of tables: "
            + TextStyle.END
            + f"{n_tables} \n \n"
        )

        summary = []
        rel_summary = []
        for table, table_details in self.schema.items():
            if self._metas[table] is not None:
                pk = table_details.primary_keys
                pk_characteristics = _get_pk_summary(
                    table_details.primary_keys, table) if len(pk) else ''

                fk = [key.column for key in table_details.foreign_keys]
                fk_characteristics = _get_fk_summary(fk, table)

                for fks in table_details.foreign_keys:
                    rel_item = {
                        'Table': fks.table,
                        'Column': fks.column,
                        'Parent Table': fks.parent_table,
                        'Parent Column': fks.parent_column,
                        'Relation Type': fks.relation_type.value,
                        # 'Rows with no children': _format_count(self.zero_children_relations[fks.table].get(fks, 0), self[fks.parent_table].summary['nrows']),
                        # 'Rows with no parents': _format_count(self.zero_parents_relations[fks.table].get(fks, 0), self[fks.table].summary['nrows']),
                    }
                    rel_summary.append(rel_item)

                table_summary = {"Table name": table,
                                 "# cols": self[table].ncols,
                                 "# nrows": self[table].summary['nrows'],
                                 "Primary keys": pk,
                                 "Foreign keys": fk if len(fk) else '',
                                 "PK characteristics": pk_characteristics if len(pk_characteristics) else '',
                                 "FK characteristics": fk_characteristics if len(fk_characteristics) else '',
                                 "Notes": ""}
            else:
                table_summary = {"Table name": table,
                                 "# cols": "",
                                 "# nrows": "",
                                 "Primary keys": "",
                                 "Foreign keys": "",
                                 "PK characteristics": "",
                                 "FK characteristics": "",
                                 "Notes": "The Metadata for this table has not been requested yet."}
            summary.append(table_summary)

        str_repr += DataFrame(summary).to_string()

        str_repr += ("\n \n"
                     + TextStyle.BOLD
                     + "Relations Summary "
                     + TextStyle.END
                     + "\n"
                     )
        str_repr += (
            TextStyle.BOLD
            + "Number of relations: "
            + TextStyle.END
            + f"{len(rel_summary)} \n \n"
        )
        str_repr += DataFrame(rel_summary).to_string()

        if self.warnings:
            str_repr += f"\n \n{TextStyle.BOLD}Warnings Summary{TextStyle.END}\n"
            for k, warning in self.warnings.items():
                str_repr += f"{TextStyle.BOLD}{k}: {TextStyle.END}"
                str_repr += f"{warning.description}\n"
        else:
            str_repr += f"\n \n{TextStyle.BOLD} No warnings Found {TextStyle.END}\n"

        return str_repr

    def save(self, path: str):
        "Creates a pickle of the metadata object stored in the provided path."
        try:
            # The deferred request endpoint must be forgotten before saving because it can't be pickled.
            #   Consequently, after the load no new tables can be fetched.
            #   As such, all needed tables must be fetched before saving the object.
            self.compute()
            self._deferred_request_endpoint = None
            # Saving NameTuple as a dict to pickle the object
            with open(path, "wb") as handle:
                dump(self, handle, HIGHEST_PROTOCOL)
        except FileNotFoundError:
            raise Exception(
                f"The directory implied in the provided path: '{path}', could not be found. Please save \
                in an existing folder."
            )

    @staticmethod
    def load(path: str) -> Metadata:
        "Loads a metadata object from a path to a pickle."
        try:
            with open(path, "rb") as handle:
                metadata = load(handle)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"The path: '{path}', could not be found. Please provide an existing load path."
            )
        assert isinstance(
            metadata, MultiMetadata
        ), "The provided path does not refer to a MultiMetadata object. Please \
            verify your input path."
        return metadata

    def _check_referential_integrity(self, foreign_keys: list[ForeignReference], multidataset: MultiDataset) -> float:
        def column_conversion(x): return x.astype(
            float) if is_numeric_dtype(x) else x
        relationships = {}
        valid_count, row_count, child_nkeys = {}, {}, {}
        for fk in foreign_keys:
            relationships[fk] = {
                'child': column_conversion(multidataset[fk.table].to_dask()[fk.column].dropna()),
                'parent': column_conversion(multidataset[fk.parent_table].to_dask()[fk.parent_column].dropna().unique().values)
            }
            child_nkeys[f'{fk.table}.{fk.column}'] = relationships[fk]['child'].shape[0]
            relationships[fk] = relationships[fk]['child'].astype(
                'category').cat.set_categories(relationships[fk]['parent'])
            valid_count[f'{fk.table}.{fk.column}'] = relationships[fk].dropna(
            ).shape[0]
            row_count[f'{fk.table}.{fk.column}'] = relationships[fk].shape[0]
        tasks = {
            'valid_count': valid_count,
            'row_count': row_count,
            'child_nkeys': child_nkeys,
        }
        result = DataFrame(compute(tasks)[0])
        # filter empty tables
        empty = result[result['row_count'] == 0]
        if empty.shape[0] == result.shape[0]:
            # if both tables are empty keys
            if empty['child_nkeys'].sum() == 0:
                return 1.
            # only the child table has values
            else:
                return 0.
        else:
            return result['valid_count'].sum() / result['row_count'].sum()

    def _compare_table_schema(
        self,
        table_name: str,
        other_metadata: MultiMetadata,
        check_referential_integrity: bool,
        multiset: MultiDataset | None = None,
        other_multiset: MultiDataset | None = None,
    ):
        table1_schema = self.schema[table_name]
        table2_schema = other_metadata.schema[table_name]

        def is_pk_valid(table: str, schema: TableSchema, data: Dataset | None, warnings: dict, nrows: int):
            if nrows == 0:
                return True

            if len(schema.primary_keys) == 0:
                return True
            elif len(schema.primary_keys) == 1:
                uniques = [
                    w for w in warnings["unique"]
                    if w.column in schema.primary_keys
                ]
                missings = [
                    w for w in warnings["missings"]
                    if w.column in schema.primary_keys
                ]
                is_unique = (
                    len(uniques) == 1 and
                    uniques[0].details["value"] == 1
                )
                has_missing = len(missings) > 0
                return is_unique and not has_missing

            if data is None:
                logger.warning(
                    f"Unable to evaluate table [{table}] composite primary key constraints without the table data."
                )
                return False

            vc = value_counts(data._data[schema.primary_keys])
            is_unique = vc.shape[0] == nrows
            dna = dropna(data._data[schema.primary_keys])
            has_missing = dna.shape[0] < nrows
            return is_unique and not has_missing

        table1_pk_is_valid = is_pk_valid(
            table_name,
            table1_schema,
            multiset[table_name] if multiset else None,
            self[table_name].warnings,
            self[table_name].shape[0],
        )
        table2_pk_is_valid = is_pk_valid(
            table_name,
            table2_schema,
            other_multiset[table_name] if multiset else None,
            other_metadata[table_name].warnings,
            other_metadata[table_name].shape[0],

        )
        compare_schema = {
            "current_schema": {
                "primary_keys": [{"column": pk, "is_valid": table1_pk_is_valid} for pk in table1_schema.primary_keys],
                "foreign_keys": table1_schema.foreign_keys
            },
            "reference_schema": {
                "primary_keys": [{"column": pk, "is_valid": table2_pk_is_valid} for pk in table2_schema.primary_keys],
                "foreign_keys": table2_schema.foreign_keys
            },
            "non_matching": {
                "primary_keys": [],
                "foreign_keys": []
            },

        }

        # This check is optional due to its computational cost.
        if check_referential_integrity:
            if multiset is not None and other_multiset is not None:
                for schema_name, multidataset in [("current_schema", multiset), ("reference_schema", other_multiset)]:
                    if compare_schema[schema_name]["foreign_keys"]:
                        compare_schema[schema_name]["perc_valid_foreign_keys"] = self._check_referential_integrity(
                            foreign_keys=compare_schema[schema_name]["foreign_keys"], multidataset=multidataset)
            else:
                logger.warning(
                    "Referential integrity not calculated, `MultiDataset`s were not provided.")

        # Check PKs
        for pk_name_tbl1 in table1_schema.primary_keys:
            pk_tbl1 = [pk for pk in compare_schema["current_schema"]
                       ["primary_keys"] if pk["column"] == pk_name_tbl1]
            pk_tbl2 = [pk for pk in compare_schema["reference_schema"]
                       ["primary_keys"] if pk["column"] == pk_name_tbl1]
            if len(pk_tbl2) == 0 or pk_tbl1[0]["is_valid"] != pk_tbl2[0]["is_valid"]:
                compare_schema["non_matching"]["primary_keys"].append(
                    pk_name_tbl1)

        # Check FKs
        for fk_tbl1 in table1_schema.foreign_keys:
            fk_tbl2 = None
            for fk_it in table2_schema.foreign_keys:
                if fk_it.column == fk_tbl1.column:
                    fk_tbl2 = fk_it
                    break

            # If the FK for this column doesn't exist or its characteristics are different
            if fk_tbl2 is None:
                compare_schema["non_matching"]["foreign_keys"].append(
                    fk_tbl1.column)
            elif fk_tbl1.parent_table != fk_tbl2.parent_table or \
                    fk_tbl1.parent_column != fk_tbl2.parent_column:
                compare_schema["non_matching"]["foreign_keys"].append(
                    fk_tbl2.column)

        return compare_schema

    def validate_schema(
        self,
        other_metadata: MultiMetadata,
        check_referential_integrity: bool,
        multiset: MultiDataset | None = None,
        other_multiset: MultiDataset | None = None,
    ) -> dict:
        schema_validations = {}
        schema_scores = {
            "num_primary_key_violations": 0,
            "num_foreign_key_violations": 0,
            "num_primary_key": 0,
            "num_foreign_key": 0
        }
        percentage_valid_fks_current = []
        percentage_valid_fks_reference = []

        def is_table_valid(meta: MultiMetadata, data: MultiDataset, table: str) -> bool:
            return (
                table in meta
                and table in data
                and data in meta.schema
                and meta[table] is not None
            )
        for tbl_name in self.schema.keys():
            if (
                is_table_valid(self, multiset, tbl_name)
                and is_table_valid(other_metadata, other_multiset, tbl_name)
            ):
                schema_validations[tbl_name] = self._compare_table_schema(
                    table_name=tbl_name,
                    multiset=multiset,
                    other_metadata=other_metadata,
                    other_multiset=other_multiset,
                    check_referential_integrity=check_referential_integrity
                )

                violations_pk = [pk for pk in schema_validations[tbl_name]["current_schema"]["primary_keys"]
                                 if not pk['is_valid']]

                schema_scores["num_primary_key_violations"] += 1 if len(
                    violations_pk) else 0
                schema_scores["num_primary_key"] += 1 if len(
                    other_metadata.schema[tbl_name].primary_keys) else 0

                schema_scores["num_foreign_key_violations"] += len(
                    schema_validations[tbl_name]["non_matching"]["foreign_keys"])
                schema_scores["num_foreign_key"] += len(
                    other_metadata.schema[tbl_name].foreign_keys)

                if "perc_valid_foreign_keys" in schema_validations[tbl_name]["current_schema"]:
                    percentage_valid_fks_current.append((schema_validations[tbl_name]["current_schema"]["perc_valid_foreign_keys"],
                                                         len(schema_validations[tbl_name]["current_schema"]["foreign_keys"])))

                if "perc_valid_foreign_keys" in schema_validations[tbl_name]["reference_schema"]:
                    percentage_valid_fks_reference.append((schema_validations[tbl_name]["reference_schema"]["perc_valid_foreign_keys"],
                                                           len(schema_validations[tbl_name]["reference_schema"]["foreign_keys"])))

        if check_referential_integrity:
            weighted_percentage_fks_current = 0.
            if percentage_valid_fks_current:
                weights = [p[1] for p in percentage_valid_fks_current]
                weights = weights if sum(weights) > 0 else None
                weighted_percentage_fks_current = np_average([p[0] for p in percentage_valid_fks_current],
                                                             weights=[p[1] for p in percentage_valid_fks_current])

            weighted_percentage_fks_reference = 0.
            if percentage_valid_fks_reference:
                weights = [p[1] for p in percentage_valid_fks_reference]
                weights = weights if sum(weights) > 0 else None
                weighted_percentage_fks_reference = np_average([p[0] for p in percentage_valid_fks_reference],
                                                               weights=[p[1] for p in percentage_valid_fks_reference])
            schema_scores["relationship_quality"] = 1.0 - \
                abs(weighted_percentage_fks_current -
                    weighted_percentage_fks_reference)

        validations = {"summary": schema_scores, "tables": schema_validations}
        return validations

    def is_same_schema(
        self,
        other_metadata: MultiMetadata,
        multiset: MultiDataset,
        other_multiset: MultiDataset
    ) -> bool:
        """Return True is both schema are strictly similar, else False."""
        schema = self.validate_schema(
            other_metadata=other_metadata,
            multiset=multiset,
            other_multiset=other_multiset,
            check_referential_integrity=True
        )
        for _, v in schema['tables'].items():
            if v['non_matching']['primary_keys'] or v['non_matching']['foreign_keys']:
                return False
        return True

    @staticmethod
    def _get_keys_str(pks: list[str | dict]):
        if not pks:
            return "None"
        if isinstance(pks[0], str):
            return ', '.join(pks)
        else:
            return ', '.join([f"{pk['column']} [{'VALID' if pk['is_valid'] else 'INVALID'}]" for pk in pks])

    @staticmethod
    def _get_fk_str(fk: ForeignReference):
        return f"{fk.column} -> {fk.parent_table}.{fk.parent_column} ({fk.relation_type.value.upper()})"

    def get_schema_validation_summary(
        self,
        other_metadata: MultiMetadata,
        multiset: MultiDataset,
        other_multiset: MultiDataset,
    ) -> str:
        schema_validations = self.validate_schema(
            multiset=multiset,
            other_multiset=other_multiset,
            other_metadata=other_metadata,
            check_referential_integrity=True
        )

        def _schema_repr(fk_info, fk_ix, m: MultiMetadata, zero_children: bool) -> str:
            fk = fk_info[fk_ix]
            parent_table = fk.table
            zero_rel = m.zero_parents_relations[parent_table].get(fk, 0)
            if not zero_children:
                zero_rel = m.zero_children_relations[parent_table].get(
                    fk, 0)
            return _format_count(zero_rel, m[fk.table if zero_children else fk.parent_table].summary['nrows'])

        fk_violations_count: int = schema_validations["summary"]["num_primary_key_violations"]
        fk_violations_percentage: float = 100 * float(schema_validations["summary"]
                                                      ["num_primary_key_violations"]) / schema_validations["summary"]["num_primary_key"]

        pk_violations_count: int = schema_validations["summary"]["num_foreign_key_violations"]
        if schema_validations["summary"]["num_foreign_key"] > 0:
            pk_violations_percentage: float = 100 * float(schema_validations["summary"]
                                                          ["num_foreign_key_violations"]) / schema_validations["summary"]["num_foreign_key"]
        else:
            pk_violations_percentage: float = 0.0

        str_repr = TextStyle.BOLD + "Schema Validation Summary\n\n" + TextStyle.END

        str_repr += TextStyle.BOLD + "Number of Primary Key Violations: " + \
            TextStyle.END + \
            f"{fk_violations_count} ({fk_violations_percentage:.2f}%)\n"
        str_repr += TextStyle.BOLD + "Number of Foreign Key Violations: " + \
            TextStyle.END + \
            f"{pk_violations_count} ({pk_violations_percentage}%)\n"
        if "relationship_quality" in schema_validations["summary"]:
            str_repr += TextStyle.BOLD + "Relationship Quality: " + TextStyle.END + \
                f'{schema_validations["summary"]["relationship_quality"] * 100:.0f}' + "%\n"
        str_repr += "\n\n"

        for tbl_name, tbl_info in schema_validations["tables"].items():
            str_repr += TextStyle.BOLD + f"Table {tbl_name}\n" + TextStyle.END
            str_repr += TextStyle.BOLD + "\n\tPrimary Keys\n" + TextStyle.END
            str_repr += f"\t\tCurrent Schema:   {self._get_keys_str(tbl_info['current_schema']['primary_keys'])}\n"
            str_repr += f"\t\tReference Schema: {self._get_keys_str(tbl_info['reference_schema']['primary_keys'])}\n"

            for fk_ix in range(max(len(tbl_info['current_schema']['foreign_keys']), len(tbl_info['reference_schema']['foreign_keys']))):
                str_repr += TextStyle.BOLD + \
                    f"\n\tForeign Key {fk_ix + 1}\n" + TextStyle.END
                if fk_ix < len(tbl_info['current_schema']['foreign_keys']):
                    str_repr += f"\t\tCurrent Schema:   {self._get_fk_str(tbl_info['current_schema']['foreign_keys'][fk_ix])}\n"
                if fk_ix < len(tbl_info['reference_schema']['foreign_keys']):
                    str_repr += f"\t\tReference Schema: {self._get_fk_str(tbl_info['reference_schema']['foreign_keys'][fk_ix])}\n"

                '''
                if tbl_info['current_schema']['foreign_keys'][fk_ix].relation_type == RelationType.ONE_TO_MANY:
                    str_repr += "\t\tRows with no children:\n"
                    if fk_ix < len(tbl_info['current_schema']['foreign_keys']):
                        repr = _schema_repr(
                            tbl_info['current_schema']['foreign_keys'], fk_ix, m=self, zero_children=False)
                        str_repr += f"\t\t\tCurrent Schema:   {repr}\n"
                    if fk_ix < len(tbl_info['reference_schema']['foreign_keys']):
                        repr = _schema_repr(
                            tbl_info['reference_schema']['foreign_keys'], fk_ix, m=other_metadata, zero_children=False)
                        str_repr += f"\t\t\tCurrent Schema:   {repr}\n"

                    str_repr += "\t\tRows with no parents:\n"
                    if fk_ix < len(tbl_info['current_schema']['foreign_keys']):
                        repr = _schema_repr(
                            tbl_info['current_schema']['foreign_keys'], fk_ix, m=self, zero_children=True)
                        str_repr += f"\t\t\tCurrent Schema:   {repr}\n"
                    if fk_ix < len(tbl_info['reference_schema']['foreign_keys']):
                        repr = _schema_repr(
                            tbl_info['reference_schema']['foreign_keys'], fk_ix, m=other_metadata, zero_children=True)
                        str_repr += f"\t\t\tCurrent Schema:   {repr}\n"
                '''
            if "perc_valid_foreign_keys" in tbl_info['current_schema'] or "perc_valid_foreign_keys" in tbl_info['reference_schema']:
                str_repr += TextStyle.BOLD + "\n\tPercentage of Valid Foreign Keys\n" + TextStyle.END
            if "perc_valid_foreign_keys" in tbl_info['current_schema']:
                str_repr += f"\t\tCurrent Schema: {tbl_info['current_schema']['perc_valid_foreign_keys'] * 100:.0f}%\n"
            if "perc_valid_foreign_keys" in tbl_info['reference_schema']:
                str_repr += f"\t\tReference Schema: {tbl_info['reference_schema']['perc_valid_foreign_keys'] * 100:.0f}%\n"

            str_repr += TextStyle.BOLD + "\n\tNon-Matching Primary Keys: " + TextStyle.END + \
                self._get_keys_str(
                    tbl_info["non_matching"]["primary_keys"]) + "\n"
            str_repr += TextStyle.BOLD + "\n\tNon-Matching Foreign Keys: " + TextStyle.END + \
                self._get_keys_str(
                    tbl_info["non_matching"]["foreign_keys"]) + "\n\n\n"
        return str_repr

    def set_table_dataset_type(
        self,
        table_name: str,
        dataset_type: str | DatasetType,
        dataset_attrs: dict | None = None
    ):
        """Update table's metadata dataset type.

        Args:
            table_name (str): Table that will have the dataset type updated
            dataset_type (str | DatasetType): new dataset type
            dataset_attrs (dict | None, optional): Dataset attrs for TIMESERIES dataset. Defaults to None.

        Raises:
            KeyError: when MultiMetadata does not contain the any table named as {table_name}
        """
        if table_name not in self._metas:
            raise KeyError(f"MultiMetadata does not have table [{table_name}]")
        self[table_name].set_dataset_type(
            dataset_type=dataset_type, dataset_attrs=dataset_attrs)

    def set_table_dataset_attrs(
        self,
        table_name: str,
        sortby: str | list,
        entities: str | list | None = None
    ):
        """Update table's metadata dataset attributes.

        Args:
            table_name (str): Table that will have the dataset attributes updated
            sortby (str | List[str]): Column(s) that express the temporal component
            entities (str | List[str] | None, optional): Column(s) that identify the entities. Defaults to None

        Raises:
            KeyError: when MultiMetadata does not contain the any table named as {table_name}
        """
        if table_name not in self._metas:
            raise KeyError(f"MultiMetadata does not have table [{table_name}]")
        self[table_name].set_dataset_attrs(sortby=sortby, entities=entities)

