"""Synthesizer model."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from os import getenv
from pathlib import Path
from random import choices, shuffle
from typing import Callable, Dict, Tuple
from warnings import warn

from dask import compute
from dill import dump as pdump
from dill import load as pload
from numpy import ceil as npceil
from numpy import nan as npnan
from numpy import repeat
from numpy.random import choice
from pandas import DataFrame as pdDataFrame
from pandas import RangeIndex as pdRangeIndex
from pandas import concat
from pandas.api.types import is_integer_dtype

from ydata.connectors.storages.rdbms_connector import RDBMSConnector
from ydata.core.connectors import WriteMode
from ydata.dataset import Dataset, DatasetType, MultiDataset
from ydata.dataset.schemas import DatasetSchema as Schema
from ydata.dataset.schemas.datasets_schema import ForeignReference, MultiTableSchema, TableSchema
from ydata.dataset.utils import humanize_dtypes
from ydata.metadata.column import Column
from ydata.metadata.metadata import Metadata
from ydata.metadata.multimetadata import MultiMetadata
from ydata.metadata.utils import drop_null
from ydata.preprocessors.methods.anonymization import AnonymizerConfigurationBuilder, AnonymizerType
from ydata.preprocessors.preprocess_methods import AnonymizerEngine
from ydata.synthesizers.base_model import BaseModel, DATATYPE_MAPPING
from ydata.synthesizers.conditional import ConditionalFeature
from ydata.synthesizers.faker import FakerSynthesizer
from ydata.synthesizers.logger import synthlogger_config
from ydata.synthesizers.multitable.calculated_features import (apply_table_calculated_features,
                                                               drop_table_calculated_features_columns,
                                                               init_calculated_features, validate_calculated_features)
# from ydata.datascience.common import EncoderType
from ydata.synthesizers.multitable.encoder import EncoderFabric, EncoderType
from ydata.synthesizers.multitable.preprocessing import (add_string_missing_values_placeholder, get_tables_to_encode,
                                                         rename_column_names, update_references_to_attribute_tables)
from ydata.synthesizers.multitable.util import (get_expected_size, get_relationship, get_table_relationships,
                                                get_tables_to_persist, is_parent_persisted, merge_tables,
                                                replace_for_valid_keys, topological_sort)
from ydata.synthesizers.regular import RegularSynthesizer
from ydata.synthesizers.timeseries import TimeSeriesSynthesizer
from ydata.utils.data_types import DataType, VariableType
from ydata.utils.misc import log_time_factory
from ydata.utils.random import RandomSeed
from ydata.utils.logger import SDKLogger
from ydata._licensing import synthesizer_sample


metrics_logger = SDKLogger(name="Metrics logger")

# Define here the logging status and definition
logger = synthlogger_config(verbose=getenv(
    "VERBOSE", "false").lower() == "true")


SYNTHESIZER = type[
    RegularSynthesizer |
    TimeSeriesSynthesizer |
    FakerSynthesizer
]


@dataclass
class Component:
    tables: list[str]
    relations: dict
    visitor_sequence: list
    merged_order: dict
    schema: dict
    synthesizer: BaseModel
    merged_rows: int
    num_entities: int


@dataclass
class _AnonymizationData:
    anonymized_columns: dict[str, list[str]]
    key_mappings: dict[str, dict[str, dict]]

    def __init__(self):
        self.anonymized_columns = defaultdict(list)
        self.key_mappings = {}
        self.builders: dict[str, AnonymizerConfigurationBuilder] = {}


class SAMPLE_METHOD(Enum):
    ORIGINAL = 1
    RELATION_BASED = 2


class MultiTableSynthesizer(BaseModel):
    # DataTypes that we are able to synthesize
    __name__ = "MultiTableSynthesizer"
    _SUPPORTED_DTYPES = [DataType.NUMERICAL, DataType.STR,
                         DataType.CATEGORICAL, DataType.DATE]

    """
    A synthesizer for generating synthetic relational databases.

    The `MultiTableSynthesizer` is designed to learn from real **relational database schemas**
    and generate new synthetic databases while preserving relationships and referential integrity.
    It supports **multi-table synthesis**, anonymization, and encoding strategies to ensure
    high-quality synthetic data generation.

    ## Key Features:
    - **Multi-Table Synthesis**: Learns from real RDBMS schemas and generates new relational data.
    - **Schema-Aware Generation**: Captures foreign key relationships and data dependencies.
    - **Anonymization Support**: Provides an option for data masking and anonymization.
    - **Referential integrity**: Maintains mapping and references across keys in different tables

    ## Example Usage:
    ```python
    from ydata.synthesizers import MultiTableSynthesizer

    # Step 1: Train the model on a Relational database
    # Initialize a synthesizer
    synth = MultiTableSynthesizer()

    # Train on a relational database
    synth.fit(multidataset, multimetadata)

    # Step 2: Generate synthetic data
    # Generate new synthetic data and write it to a destination RDBMS
    synthetic_db = synth.sample(1.0, connector=connector_rdbms)

    # Step 3: Save the trained model
    synth.save("model.pkl")

    # Step 4: Load the trained model later
    loaded_synth = RegularSynthesizer.load("model.pkl")
    ```
    """

    def __init__(
        self, *, tmppath: str | Path | None = None
    ):
        super().__init__(tmppath=tmppath)
        self.schema = None
        self.components = []
        self.tables_schemas = None
        self.tables_columns = None
        self.mt_anonymize = None
        self.sample_method = SAMPLE_METHOD.RELATION_BASED
        self.tables_dataset_attrs = None
        self._key_values = {}
        self._encoded_columns: dict[str, list[str]] = {}

    @property
    def SUPPORTED_DTYPES(self):  # noqa: N802
        return self._SUPPORTED_DTYPES

    def _postprocess_missing_values(self, tables_df):
        """Invert transform `MISSING` back to `np.nan`."""
        for n, t in tables_df.items():
            tables_df[n] = t.replace("MISSING", npnan)
        return tables_df

    def _postprocess_NA_values(self, sample_tables, tables_schemas):
        """Replace `<NA>` from pd.NA as dask does not support it."""
        for n in sample_tables.keys():
            for c, k in tables_schemas[n].items():
                if c in sample_tables[n].columns and k in [VariableType.FLOAT, VariableType.INT]:
                    # Dask does not support <NA> and will infer string instead of None
                    sample_tables[n][c] = sample_tables[n][c].replace(
                        '<NA>', None)
        return sample_tables

    def _postprocess_NA_categoricals(self, table_data: pdDataFrame):
        """Replace `nan` values that were converted to string back to
        `None`."""
        for col in table_data.columns:
            if 'nan' in table_data[col].unique():
                table_data[col] = table_data[col].replace('nan', None)
        return table_data

    def __select_synthesizer(
        self, table: str
    ) -> RegularSynthesizer | FakerSynthesizer:
        if self.is_attribute_table(table):
            model = FakerSynthesizer()
            model._anonymize_categoticals = True
        else:
            model = RegularSynthesizer(tmppath=self.tmppath)
            model.is_multitable = True
            model._bypass_min_rows_limit = True
        return model

    def __create_timeseries_synth(
        self, metadata: MultiMetadata, table: str
    ) -> Tuple[BaseModel, Dict]:
        """Placeholder method of how the timeseries synthesizer used to be
        created."""
        dataset_attrs = {}
        if metadata.dataset_type[table] == DatasetType.TIMESERIES:
            synth_cls = TimeSeriesSynthesizer
            # At the moment we don't support sortbykey on multiple cols
            if "sortbykey" in metadata._dataset_attrs[table]:
                dataset_attrs["sortbykey"] = f"{table}.{metadata._dataset_attrs[table]['sortbykey']}"
            if "entities" in metadata._dataset_attrs[table]:
                if "entities" not in dataset_attrs:
                    dataset_attrs["entities"] = []
                for ent_col_id in metadata._dataset_attrs[table]['entities']:
                    dataset_attrs["entities"].append(
                        f"{table}.{ent_col_id}")
        return synth_cls, dataset_attrs

    @staticmethod
    def __merge_metadata(
        merged: pdDataFrame,
        dataset_type: str,
        metadata: MultiMetadata,
        dataset_attrs: Dict,
    ):
        merged_m = Metadata(dataset_attrs=dataset_attrs)
        if dataset_type == "timeseries":
            merged_m._Metadata__validate_dataset_attrs(
                dataset_attrs=dataset_attrs)
        merged_m._Metadata__columns = {}
        for col in merged.columns:
            names = col.split("|")[0].split(".", 1)
            col_name = names[-1]
            table_name = names[0]
            if col_name in metadata._metas[table_name].columns:
                merged_m._Metadata__columns[col] = metadata._metas[table_name].columns[col_name]

        merged_m._Metadata__warnings = {}
        merged_m._nrows = merged.shape[0]
        merged_m._metadata["nrows"] = merged.shape[0]
        merged_m._metadata["summary"] = {
            "nrows": merged.shape[0],
            "iscategorical": {},
            "missings": {},
            "skewness": {},
            "string_len": {},
            "cardinality": {},
            "unique_counts": {},
        }
        return merged_m

    def _init_anonymizer_configuration(self) -> dict:
        # Automatically anonymize primary keys if required
        ANON_PK = 'anonymize_primary_keys'
        anonymize_all = False
        if ANON_PK in self.mt_anonymize:
            anonymize_all = True
            del self.mt_anonymize[ANON_PK]
        for t, v in self.schema.items():
            pks = v.primary_keys
            anonymize_table = False
            for pk in pks:
                if ANON_PK in self.mt_anonymize.get(t, {}):
                    anonymize_table = True
                    del self.mt_anonymize[t][ANON_PK]
                if pk in self.mt_anonymize.get(t, {}):
                    continue  # a specific anonymizer was already defined
                if anonymize_all or anonymize_table:  # Anonymize only if asked globally or at the table level
                    if t not in self.mt_anonymize:
                        self.mt_anonymize[t] = {}

                    self.mt_anonymize[t][pk] = AnonymizerType.INT

    def _create_metadata(
        self,
        df: pdDataFrame,
        dataset: Dataset,
        metadata: MultiMetadata,
        dataset_attrs: dict,
        synth: SYNTHESIZER
    ):
        dataset_type = "timeseries" if isinstance(
            synth, TimeSeriesSynthesizer) else "tabular"
        meta = Metadata(
            dataset_attrs=dataset_attrs,
            dataset_type=dataset_type,
            pairwise_metrics=False
        )
        meta._is_multitable = True
        meta(
            dataset,
            dataset_attrs=dataset_attrs,
            dataset_type=dataset_type
        )

        meta_aux = self.__merge_metadata(
            df, dataset_type, metadata, dataset_attrs
        )
        for k, v in meta_aux.columns.items():
            if v.vartype == VariableType.DATETIME:
                meta.columns[k] = Column(
                    k, DataType.DATE, VariableType.DATETIME
                )

            table_name, col_name = k.split('.', 1)
            datatype = metadata[table_name].columns[col_name].datatype
            meta.columns[k].datatype = datatype

        return meta

    def _get_child_columns(self, table: str, column: str, child: str):
        schema = self.schema.get(child, TableSchema())
        cols = []
        for fk in schema.foreign_keys:
            if table == fk.parent_table and column.split(".", 1)[-1] == fk.parent_column:
                cols.append(fk.column)
        return cols

    def _update_references_to_anonymized_columns(
        self,
        table: str,
        table_data: pdDataFrame,
        anonymization_data: _AnonymizationData,
    ):
        table_schema = self.schema.get(table, TableSchema())
        for fk in table_schema.foreign_keys:
            if fk.parent_table in anonymization_data.key_mappings:
                parent_mapping = anonymization_data.key_mappings.get(
                    fk.parent_table)
                if fk.parent_column in parent_mapping:
                    table_data[fk.column] = table_data[fk.column].map(
                        parent_mapping[fk.parent_column],
                        na_action="ignore",  # propagates the NaN values
                    )
        return table_data

    def _update_pk_references(
        self,
        table: str,
        column: str,
        tables_df: dict[str, pdDataFrame],
        mapping: dict[str, list],
        anonymized_tables: dict[str, list[str]],
    ):
        """Recursivelly updates all references for an anonymized PK."""
        order = [t for t, _ in self._tables_order]
        tid = order.index(table)
        for child, _ in self._tables_order[tid:]:
            child_columns = self._get_child_columns(table, column, child)
            anonymized_tables[child].extend(child_columns)
            for col in child_columns:
                tables_df[child][col] = tables_df[child][col].map(
                    mapping)
                self._update_pk_references(
                    child, col, tables_df, mapping, anonymized_tables)

    @log_time_factory(logger)
    def anonymize_table_pks(self, table: str, table_df: pdDataFrame) -> pdDataFrame:
        config = {
            k: v for k, v in self.mt_anonymize.get(table, {}).items()
            if k in table_df.columns and k in self.schema[table].primary_keys
        }
        config = AnonymizerEngine.process_config(config)
        builder = AnonymizerConfigurationBuilder(config)
        for anonymizer in builder.get_config().values():
            table_df[anonymizer.cols] = anonymizer.get_anonymizer()(
                table_df[anonymizer.cols],
                {
                    col: len(table_df[col])
                    if col in self.schema[table].primary_keys
                    else self.tables_cardinality[table][col]
                    for col in anonymizer.cols
                    if col in table_df.columns
                },
                **anonymizer.params
            )
        return table_df

    @log_time_factory(logger)
    def _anonymize_table(
        self,
        table: str,
        table_df: pdDataFrame,
        anonymization_data: _AnonymizationData,
    ) -> Tuple[pdDataFrame, dict, dict]:
        builder = anonymization_data.builders[table]
        for anonymizer in builder.get_config().values():
            og_values = {}
            anonymization_data.anonymized_columns[table].extend(
                anonymizer.cols)
            for col in anonymizer.cols:
                if col in self.schema[table].primary_keys and col in table_df.columns:
                    og_values[col] = table_df[col].value_counts(
                    ).index.to_list()

            table_df[anonymizer.cols] = anonymizer.get_anonymizer()(
                table_df[anonymizer.cols],
                {
                    col: len(table_df[col])
                    if col in self.schema[table].primary_keys
                    else self.tables_cardinality[table][col]
                    for col in anonymizer.cols
                    if col in table_df.columns
                },
                **anonymizer.params
            )

            anonymization_data.key_mappings[table] = {}
            for col in og_values:
                anon_values = table_df[col].value_counts().index.to_list()
                mapping = {k: v for k, v in zip(og_values[col], anon_values)}
                anonymization_data.key_mappings[table][col] = mapping

        return table_df, anonymization_data

    def update_anonymized_columns_metadata(
        self,
        table: str,
        metadata: Metadata,
        dataset_schema,
        anonymization_data: _AnonymizationData
    ):
        """update metadata types after anonymization."""
        for col in anonymization_data.anonymized_columns[table]:
            dataset_vartype = dataset_schema[col].vartype
            metadata_vartype = metadata.columns[col].vartype
            if metadata_vartype == dataset_vartype:
                continue
            else:
                metadata.columns[col]._vartype = dataset_vartype

            updates = {}
            if metadata.columns[col].datatype == DataType.CATEGORICAL:
                updates[col] = DataType.CATEGORICAL
            else:
                if dataset_vartype == VariableType.INT or dataset_vartype == VariableType.FLOAT:
                    updates[col] = DataType.NUMERICAL
                elif dataset_vartype == VariableType.DATE or dataset_vartype == VariableType.DATETIME:
                    updates[col] = DataType.DATE
                else:
                    updates[col] = DataType.STR
            metadata.update_datatypes(updates)

    def _anonymize_single_table(
        self,
        table: str,
        table_data: Dataset,
        table_metadata: Metadata,
        anonymization_data: _AnonymizationData,
    ):
        """Anonymize tables following the mt_anonymize doc.

        Args:
            table (str): table name
            table_data (Dataset): dataset to be anonymized
            table_metadata (Metadata): table_data metadata.
                Obs.: The column types may be update depending of the anonymization type.
            anonymization_data (_AnonymizationData): auxiliary data for anonymization.
        """
        table_df = table_data.to_pandas()
        if table not in self.mt_anonymize:
            return table_df, table_metadata, anonymization_data

        table_df, anonymization_data = self._anonymize_table(
            table=table,
            table_df=table_df,
            anonymization_data=anonymization_data,
        )
        logger.info(
            f"[SYNTHESIZER] - Anonymizing table {table} columns: {anonymization_data.anonymized_columns.get(table, [])}")

        self._update_references_to_anonymized_columns(
            table=table,
            table_data=table_df,
            anonymization_data=anonymization_data,
        )
        dtypes, _ = humanize_dtypes(table_df.dtypes.to_dict())
        self.update_anonymized_columns_metadata(
            table=table,
            metadata=table_metadata,
            anonymization_data=anonymization_data,
            dataset_schema={
                col: Schema(column=col, vartype=VariableType(v))
                for col, v in dtypes.items()
            },
        )

        return table_df, table_metadata, anonymization_data

    def _fit_initialization(
        self,
        X: MultiDataset,
        metadata: MultiMetadata,
        anonymize: dict | None = None,
    ) -> tuple[MultiDataset, MultiMetadata]:
        """Filter multi dataset and multi metadata of constant and empty
        tables. Also initialize shorthands of schema and metadata fields. And
        remove int anonymization configuration for PKs since it is redundant.

        Args:
            X (MultiDataset): database to fit
            metadata (MultiMetadata): database metadata
            anonymize (dict | None, optional): anonymization configuration. Defaults to None.

        Returns:
            tuple[MultiDataset, MultiMetadata]: database, metadata
        """
        self.mt_anonymize = anonymize if anonymize is not None else {}
        self.schema = X.schema
        self.references = defaultdict(list[ForeignReference])

        for table_name, table in self.schema.items():
            for fk in table.foreign_keys:
                self.references[fk.parent_table].append(fk)
            for pk in table.primary_keys:
                table_anonymize_config = self.mt_anonymize.get(table_name, {})
                # Remove anonymizer INT on pk as it is not necessary by construction
                if pk in table_anonymize_config and table_anonymize_config[pk] == AnonymizerType.INT:
                    del self.mt_anonymize[table_name][pk]

        self.tables_schemas = {t: X[t].schema for t in X.keys()}
        self.total_cols = sum([t.ncols for t in X._datasets.values()])
        self.tables_nrows = {t: metadata[t].summary['nrows'] for t in X.keys()}
        self.missing_values = {
            t: metadata[t].summary['missings'] for t in X.keys()}
        self.tables_cardinality = {
            t: metadata[t].summary['cardinality'] for t in X.keys()}

        # only used for timeseries synthesizers
        self.tables_dataset_type = metadata.dataset_type
        self.tables_dataset_attrs = metadata._dataset_attrs
        self.cardinality_dist = {}

        # metadata will be updated internally so a copy is needed to not alter the original metadata
        metadata = deepcopy(metadata)
        # identify empty tables and remove it from tables and metadata
        self.empty_tables = {t: v.to_pandas() for t, v in X.items(
        ) if metadata[t].summary['nrows'] in [0, 1]}
        X = X[[k for k in X.keys() if k not in self.empty_tables.keys()]]
        for k in self.empty_tables.keys():
            del metadata[k]

        # identify constant tables and remove it from tables and metadata
        self.constant_tables = {t: v for t, v in X.items() if set(
            v.columns) == set([c.column for c in metadata[t].warnings['constant']])}
        X = X[[k for k in X.keys() if k not in self.constant_tables.keys()]]
        for k in self.constant_tables.keys():
            del metadata[k]
        return X, metadata

    def _get_attribute_tables(
        self,
        table_list: list[str],
        attribute_tables: list[str] | set[str] | str | None,
    ) -> set[str]:
        if attribute_tables is None:
            return set()
        if isinstance(attribute_tables, str):
            attribute_tables = set([attribute_tables])
        if isinstance(attribute_tables, list):
            attribute_tables = set(attribute_tables)

        invalid_tables = [
            table
            for table in attribute_tables
            if table not in table_list
        ]

        if invalid_tables:
            warn(
                f"The following attribute tables do not exist in the database schema {invalid_tables}. Invalid tables were ignored."
            )

        attribute_tables = [
            table
            for table in attribute_tables
            if table in table_list
        ]

        return attribute_tables

    def create_synthesizers(self):
        synthesizers = {}
        for table, previous_table in self._tables_order:
            model = self.__select_synthesizer(table)

            synthesizers[table] = {
                "previous_table": previous_table,
                "model": model,
                "dataset_attrs": {},
            }
        return synthesizers

    def _get_topological_order(self, tables: list[str]):
        digraph = {k: set() for k in tables}
        for child, parent, _ in self.relationships:
            digraph[parent].add(child)
        return topological_sort(digraph)

    def is_attribute_table(self, table: str) -> bool:
        """Check if table is an attribute table.

        Args:
            table (str): table name

        Returns:
            bool: True if table is an attribute table, false otherwise.
        """
        return table in self.attribute_tables

    def _fit_attribute_tables(
        self,
        metadata: MultiMetadata,
        synthesizers: dict[str, SYNTHESIZER]
    ):
        """Fit attribute tables models.

        Args:
            metadata (MultiMetadata): Database metadata
            synthesizers (dict[str, SYNTHESIZER]): collection of table synthesizers.
        """
        for table, synth in synthesizers.items():
            if self.is_attribute_table(table):
                synth["model"].fit(metadata[table])

    def _anonymize_merged_table_pks(self, df, tables_df, metadata, table, previous_table, relationship):
        anonymized_pks = [
            f"{previous_table}.{pk}" for pk in self.schema[previous_table].primary_keys
            if pk in self.mt_anonymize.get(previous_table, {})
        ]
        if anonymized_pks:
            df = df.rename(columns={
                col: col.split(".", 1)[-1] for col in df.columns
                if col.startswith(f"{previous_table}.")
            })

            df = df.rename(columns={
                col: f"{previous_table}.{col}" for col in df.columns
                if not col.startswith(f"{table}.")
            })
            # replace fks
            if isinstance(relationship["left_on"], str):
                left_table = relationship["left_on"].split(".", 1)[0]
            else:
                left_table = relationship["left_on"][0].split(".", 1)[0]
            if previous_table == left_table:
                df[relationship["right_on"]] = df[relationship["left_on"]]
            else:
                df[relationship["left_on"]] = df[relationship["right_on"]]

            # update table df
            tables_df[table] = df[[
                col for col in df.columns
                if col.startswith(f"{table}.")
            ]].drop_duplicates().reset_index(drop=True)

            # update table metadata
            if table in self.mt_anonymize:
                self.update_anonymized_columns_metadata(
                    table=table,
                    metadata=metadata[table],
                    dataset=Dataset(tables_df[table].rename(
                        columns={
                            col: col.split(".", 1)[-1]
                            for col in tables_df[table].columns
                        })
                    )
                )

    def _get_joint_table_data(
        self,
        tables_df: dict[str, pdDataFrame],
        table: str,
        previous_table_list: list[str],
        limit: int
    ) -> pdDataFrame:
        df = tables_df[table]

        if len(previous_table_list) == 1:
            relationship = get_relationship(
                self.relationships,
                table,
                previous_table_list[0],
            )
            logger.debug(
                f"Merging tables: [{relationship}]")
            df = merge_tables(relationship, tables_df, table)
        else:
            for previous_table in previous_table_list:
                relationship = get_relationship(
                    self.relationships,
                    table,
                    previous_table,
                )
                if relationship:
                    # FIXME multiple references to the same column now fails likelly because of the name conflict in the merged table
                    # FIXME or because the column was not correctly identified as joint and the parent was not encoded because of it
                    logger.debug(
                        f"Merging table: [{relationship}]")
                    encoded_columns = []
                    if previous_table in self._encoded_columns:
                        encoded_columns = self._encoded_columns[previous_table]
                    _, keys = self._get_related_keys(table, previous_table)
                    keys = keys if isinstance(keys, list) else [keys]
                    columns_to_merge = {
                        previous_table: [
                            f"{previous_table}.{c}" for c in keys + encoded_columns]
                    }
                    df = merge_tables(
                        relationship,
                        {
                            table: df,
                            previous_table: tables_df[previous_table]
                        },
                        table,
                        columns_to_merge
                    )

                    df = df.drop(
                        columns=[f"{previous_table}.{c}" for c in keys])

        logger.debug(f"Merged table size {df.shape}")
        if df.shape[0] > limit:
            df = df.sample(n=limit)
            logger.debug(
                f"Limiting the merged table size. New size {df.shape}")
        return df

    def _encode_table(
        self,
        table: str,
        table_data: pdDataFrame,
        table_metadata: Metadata,
        encoder_type: EncoderType = EncoderType.BIRCH,
    ):
        if (
            table in self._encoded_columns
            or table in self.empty_tables
            or table in self.constant_tables
        ):
            return

        logger.info(f"[SYNTHESIZER] - Encoding table {table}")
        encoder = EncoderFabric.create(encoder_type)
        self._encoded_columns[table] = encoder.encoded_columns
        table_data[f"{table}.__cluster_id__"] = encoder.fit_predict(
            table_data,
            metadata=table_metadata,
            schema=self.schema.get(table, TableSchema()),
            table=table,
        )

    @log_time_factory(logger)
    def fit(
        self,
        X: MultiDataset,
        metadata: MultiMetadata,
        anonymize: dict | None = None,
        limit: int = 50_000_000,
        calculated_features: list[dict[str, str |
                                       Callable | list[str]]] | None = None,
        attribute_tables: list[str] | set[str] | str | None = None,
        random_state: RandomSeed = None,
        encoder_type: EncoderType = EncoderType.BIRCH,
    ):
        """Fit a MultiTable Synthesizer instance.

        The synthesizer operates over a denormalized version of the dataset.

        Args:
            X (MultiDataset): Training dataset.
            metadata (MultiMetadata): Associated metadata.
            anonymize (Optional[dict]): Defines which columns to anonymize and the anonymization method. Defaults to None.
            limit (int): Limit of rows from the denormalized dataset to use for training. Defaults to 50_000_000.
            calculated_features(Optional[ list[ dict[str, str | Callable | List[str] ]]):
                Lists the column that will be computed based on other tables/columns and the function to compute. Defaults to None.
            attribute_tables (list | set | str): collection of tables that contain static information.
            random_state: random generator or seed for the synthesizer fit
        Returns:
            MultiTableSynthesizer: trained instance of the synthesizer
        """

        metrics_logger.info(dataset=X,
                            datatype=DATATYPE_MAPPING[self.__class__.__name__],
                            method='synthesizer')

        X, metadata = self._fit_initialization(
            X=X, metadata=metadata, anonymize=anonymize)
        self.calculated_features = init_calculated_features(
            calculated_features)
        validate_calculated_features(
            data=X, calculated_features=self.calculated_features)

        tables_list = list(X)
        self.attribute_tables = self._get_attribute_tables(
            tables_list, attribute_tables)

        schema = MultiTableSchema(self.schema.dict(), tables=tables_list)
        self.relationships = get_table_relationships(schema)
        self._topological_order = self._get_topological_order(tables_list)
        self._tables_order = []
        for table in self._topological_order:
            table_schema = self.schema.get(table, TableSchema())
            parents = []
            for fk in table_schema.foreign_keys:
                if self.is_attribute_table(fk.parent_table):
                    logger.info(
                        f"removed the attribute table connection [{fk.parent_table}<-{table}]")
                elif fk.parent_table not in parents:
                    parents.append(fk.parent_table)

            order = (table, parents)
            self._tables_order.append(order)

        logger.debug("\n".join([str(t) for t in self._tables_order]))

        self._compute_cardinality_distribution(X, metadata)

        self._init_anonymizer_configuration()
        synthesizers = self.create_synthesizers()
        n_synthesizers = len(synthesizers)

        # preprocessing

        # training loop
        tables_to_encode = get_tables_to_encode(
            self.schema, self.attribute_tables)
        tables_df = {}
        fitted_tables = set()
        composite_keys = self._get_user_defined_composite_keys()

        logger.info("[MULTITABLE] - Validate anonymization data")
        anonymization_data = _AnonymizationData()
        for table in tables_list:
            config = {
                k: v for k, v in self.mt_anonymize.get(table, {}).items()
                if k in X[table].columns
            }
            config = AnonymizerEngine.process_config(config)
            builder = AnonymizerConfigurationBuilder(config)
            anonymization_data.builders[table] = builder

        for synth_i, (table, _) in enumerate(self._tables_order):
            logger.info(
                f"({synth_i+1}/{n_synthesizers}) - Fitting table: [{table}]")

            if self.is_attribute_table(table):
                # TODO should it be anonymized prior to fitting on metadata?
                synthesizers[table]["model"].fit(metadata[table])
            else:
                anonymization_data = self._fit_table(
                    X=X,
                    metadata=metadata,
                    synthesizers=synthesizers,
                    tables_df=tables_df,
                    table=table,
                    limit=limit,
                    tables_to_encode=tables_to_encode,
                    anonymization_data=anonymization_data,
                    random_state=random_state,
                )

            # clear unecessary tables
            tables_to_clear = get_tables_to_persist(
                tables=tables_list,
                sampled_tables=tables_df,
                persisted_tables=fitted_tables,
                schema=self.schema,
                composite_keys=composite_keys,
                calculated_features=self.calculated_features
            )
            if tables_to_clear:
                logger.info(
                    f"[MULTITABLE] - Clearing tables from memory [{tables_to_clear}]")
                for ttc in tables_to_clear:
                    if ttc in tables_df:
                        tables_df.pop(ttc)

                    # remove data from anonymization data
                    if table in anonymization_data.anonymized_columns:
                        anonymization_data.anonymized_columns.pop(table)
                    if table in anonymization_data.key_mappings:
                        anonymization_data.key_mappings.pop(table)

            fitted_tables |= tables_to_clear

        self.is_fitted_ = True
        self._synthesizers = synthesizers
        return self

    def _fit_table(
        self,
        X: MultiDataset,
        metadata: MultiMetadata,
        synthesizers: dict,
        tables_df: dict[str, pdDataFrame],
        table: str,
        limit: int,
        tables_to_encode: list[str],
        anonymization_data: _AnonymizationData,
        random_state: RandomSeed
    ) -> _AnonymizationData:
        synth = synthesizers[table]
        table_data = X[table]
        table_metadata = metadata[table]

        # anonymize table
        table_data, table_metadata, anonymization_data = self._anonymize_single_table(
            table=table,
            table_data=table_data,
            table_metadata=table_metadata,
            anonymization_data=anonymization_data,
        )
        table_data = rename_column_names(
            table=table,
            table_data=table_data
        )

        # handle missing values
        table_data = add_string_missing_values_placeholder(
            table=table,
            table_data=table_data,
            table_keys=self.schema.get(table, TableSchema()).get_keys(),
            table_metadata=table_metadata,
        )
        tables_df[table] = table_data

        # drop calculated features
        drop_table_calculated_features_columns(
            table=table,
            table_data=table_data,
            table_metadata=table_metadata,
            calculated_features=self.calculated_features
        )

        # update attribute tables references
        update_references_to_attribute_tables(
            table=table,
            table_data=table_data,
            metadata=metadata,
            table_schema=self.schema.get(table, TableSchema()),
            attribute_tables={k: v for k, v in synthesizers.items() if isinstance(
                v["model"], FakerSynthesizer)}
        )

        if table in tables_to_encode:
            self._encode_table(table, table_data, table_metadata)

        previous_table = synth["previous_table"]
        df = self._get_joint_table_data(
            table=table,
            tables_df=tables_df,
            previous_table_list=previous_table,
            limit=limit,
        )
        table_data = Dataset(df)
        if not synth["dataset_attrs"]:
            attr = None
        else:
            attr = synth["dataset_attrs"]
        table_metadata = self._create_metadata(
            df, table_data, metadata, attr, synth["model"])

        condition_on = []
        if previous_table:
            condition_on = [
                c for c in df.columns
                if c.split(".", 1)[0] == previous_table
            ]

        fit_params = {
            "X": table_data,
            "metadata": table_metadata,
            "condition_on": condition_on,
            "random_state": random_state,
        }
        if "dataset_attrs" in synth["dataset_attrs"]:
            fit_params["dataset_attrs"] = synth["dataset_attrs"]
        else:
            # holdout_size is only present in regular synthesizer
            fit_params["holdout_size"] = 0

        synth["model"].fit(**fit_params)
        return anonymization_data

    def _sample_cardinality_distribution(self, distribution, n_keys, expected_size, tolerance: float = 0.05):
        n_cards = choices(
            distribution.index,
            weights=distribution.to_list(),
            k=n_keys
        )

        def get_cardinality_size(card, reduction_factor):
            if card == 0:
                return 0
            return npceil(card * reduction_factor)

        cards_size = sum(n_cards)
        if cards_size <= expected_size * (1 + tolerance):
            return n_cards
        else:
            reduction_factor = expected_size / cards_size
            n_cards = [
                get_cardinality_size(card, reduction_factor)
                for card in n_cards
            ]
        return n_cards

    @staticmethod
    def _add_empty_rows_condition_df(condition_df: pdDataFrame, n_empty_rows: int):
        nrows = condition_df.shape[0]
        empty_df = pdDataFrame(
            npnan,
            index=pdRangeIndex(
                start=nrows,
                stop=nrows + n_empty_rows,
            ),
            columns=condition_df.columns
        )
        condition_df = concat(
            [condition_df, empty_df],
            axis=0
        ).reset_index(drop=True)
        condition_df = condition_df.sample(frac=1).reset_index(drop=True)
        return condition_df

    def _sample_keys(
        self,
        parent_table: pdDataFrame,
        base_table: str,
        base_column: str,
        fk: ForeignReference,
        sample_size: int
    ):
        missing = self.missing_values[base_table].get(base_column, 0)
        n_rows = self.tables_nrows[base_table]
        fraction = sample_size / n_rows

        if missing == 0:
            expected_size = sample_size
            n_orphans = 0
        else:
            expected_size = get_expected_size(
                n_rows - missing,
                fraction
            )
            n_orphans = get_expected_size(
                missing,
                fraction
            )

        # childless check
        parent_cardinality = self.tables_cardinality[fk.parent_table][fk.parent_column]
        child_cardinality = self.tables_cardinality[fk.table][fk.column]
        childless = parent_cardinality - child_cardinality
        childless = 0 if childless < 0 else childless
        n_childless = get_expected_size(childless, fraction)

        key_values = parent_table[fk.parent_column].unique()
        n_keys = len(key_values)
        if n_childless >= n_keys:
            # all children are orphans
            if expected_size == 0:
                n_childless = n_keys
            # there is only one key and parent child pairs are expected
            elif n_keys == 1:
                n_childless = 0
            else:
                # reduce the proportion of childless
                ratio = n_keys / parent_cardinality
                n_childless = get_expected_size(childless, ratio)
                # if not enough hard cap the number of childless parents
                if n_childless >= n_keys:
                    n_childless = n_keys - 1

        n_cards = self._sample_cardinality_distribution(
            distribution=self.cardinality_dist[base_table][base_column],
            n_keys=n_keys - n_childless,
            expected_size=expected_size
        )
        n_cards = n_cards + [0]*n_childless
        shuffle(n_cards)

        keys = repeat(key_values, n_cards).tolist()
        if n_orphans > 0:
            keys += [None] * n_orphans
        shuffle(keys)

        if len(keys) >= sample_size:
            return keys[:sample_size]
        else:
            keys += choice(key_values, size=sample_size-len(keys)).tolist()
            return keys

    def _get_parent_condition_df(self, condition_df, table, table_key, previous_table, previous_table_key, shapes, fraction):
        missing = self.missing_values[table].get(table_key, 0)
        if missing == 0:
            expected_size = shapes[table]
            n_orphans = 0
        else:
            expected_size = get_expected_size(
                self.tables_nrows[table] - missing,
                fraction
            )
            n_orphans = get_expected_size(
                missing,
                fraction
            )

        # get childless
        parent_cardinality = self.tables_cardinality[previous_table][previous_table_key]
        child_cardinality = self.tables_cardinality[table][table_key]
        childless = parent_cardinality - child_cardinality
        childless = 0 if childless < 0 else childless
        n_childless = get_expected_size(childless, fraction)

        key_values = condition_df[previous_table_key].unique()
        n_keys = len(key_values)
        if n_childless >= n_keys:
            # all children are orphans
            if expected_size == 0:
                n_childless = n_keys
            # there is only one key and parent child pairs are expected
            elif n_keys == 1:
                n_childless = 0
            else:
                # reduce the proportion of childless
                ratio = n_keys / parent_cardinality
                n_childless = get_expected_size(childless, ratio)
                # if not enough hard cap the number of childless parents
                if n_childless >= n_keys:
                    n_childless = n_keys - 1

        logger.debug(
            f"non missing: {expected_size}, missing {n_orphans}, n_childless {n_childless}")
        n_cards = self._sample_cardinality_distribution(
            distribution=self.cardinality_dist[table][table_key],
            n_keys=n_keys - n_childless,
            expected_size=expected_size
        )
        n_cards = n_cards + [0]*n_childless
        shuffle(n_cards)

        condition_df = condition_df.drop_duplicates(
            subset=[previous_table_key]
        ).reset_index(drop=True)
        condition_df = condition_df.loc[repeat(
            condition_df.index.values, n_cards)].reset_index(drop=True)

        if n_orphans > 0:
            condition_df = self._add_empty_rows_condition_df(
                condition_df, n_orphans)

        return condition_df

    @log_time_factory(logger)
    def get_condition_df(self, table, previous_table, sample_tables, shapes, fraction):
        table_key, previous_table_key = self._get_related_keys(
            table, previous_table)

        condition_df = sample_tables[previous_table].copy()
        is_composite = isinstance(table_key, list)
        if is_composite:
            table_key = table_key[0]
            previous_table_key = previous_table_key[0]

        condition_df = self._get_parent_condition_df(
            condition_df, table, table_key, previous_table, previous_table_key, shapes, fraction
        )

        condition_df = condition_df.rename(
            columns={col: f"{previous_table}.{col}" for col in condition_df.columns})
        return condition_df

    def _get_user_defined_composite_keys(self):
        """Create a dictionary with the user defined composite keys references.
        The dictionary is indexed by the parent table keys to ease checks after
        the table is sampled.

        Returns:
            dict: composite keys dependencies reference
        """
        inverted = defaultdict(list)

        composite = self.schema.composite_keys
        for table, ck in composite.items():
            pks = self.schema.get(
                ck['parent_table'],
                TableSchema()
            ).primary_keys
            if ck["parent_columns"][0] in pks:
                parent_reference = ck["parent_columns"][0]
                reference = ck["columns"][0]
                parent_column = ck["parent_columns"][1]
                column = ck['colummns'][1]
            else:
                parent_reference = ck["parent_columns"][1]
                reference = ck["columns"][1]
                parent_column = ck["parent_columns"][0]
                column = ck['columns'][0]
            inverted[ck['parent_table']].append({
                "parent_column": parent_column,
                "table": table,
                "column": column,
                "reference": reference,
                "parent_reference": parent_reference,
            })
        return dict(inverted)

    def _is_reference_generated(self, table: str, column: str, references: dict, sample_tables: dict[str, pdDataFrame]):
        """Check if the reference table was already generated.

        Obs.: Mappings from one reference to another is not supported.

        Args:
            table (str): _description_
            column (str): _description_
            references (dict): _description_
            sample_tables (dict[str, pdDataFrame]): _description_

        Returns:
            _type_: _description_
        """
        if table not in sample_tables:
            return False
        table_schema = self.schema.get(table, TableSchema())
        fks = [fk for fk in table_schema.foreign_keys if fk.column == column]
        if len(fks) == 0:
            # table is already generated and the column does not depend of other tables
            return True
        else:
            fk = fks[0]
            return self._is_reference_generated(fk.parent_table, fk.parent_column, references, sample_tables)

    def _handle_user_defined_composite_keys(self, composite_keys: dict, sample_tables: dict[str, pdDataFrame]):
        if (len(composite_keys) == 0):
            return

        updated = False
        for table, table_references in composite_keys.items():
            to_remove = []
            for i, ref in enumerate(table_references):
                if (
                    ref["table"] in sample_tables
                    and self._is_reference_generated(
                        table, ref["parent_column"], composite_keys, sample_tables
                    )
                ):
                    # TODO is it possible to get the reference column automatically?
                    logger.debug(f"handling composite_keys for {table}: {ref}")
                    mapping = sample_tables[table].set_index(ref["parent_reference"])[
                        ref["parent_column"]].to_dict()
                    sample_tables[ref["table"]][ref["column"]
                                                ] = sample_tables[ref["table"]][ref["reference"]].map(mapping)
                    to_remove.append(i)
                    updated = True

            # drop references
            for index in sorted(to_remove, reverse=True):
                del table_references[index]
        for table in list(composite_keys.keys()):
            if len(composite_keys[table]) == 0:
                composite_keys.pop(table)

        if updated:
            # when handling multiple composite keys that are interconnected solving one may enable the resolution of other
            self._handle_user_defined_composite_keys(
                composite_keys, sample_tables)

    def _remove_duplicated_primary_keys(
        self,
        sample_data: pdDataFrame,
        table_schema: TableSchema,
        table_name: str
    ):
        if (
            table_schema.primary_keys and
            table_schema.primary_keys[0] in sample_data.columns
        ):
            keys = table_schema.primary_keys
        else:
            keys = [
                f"{table_name}.{pk}"
                for pk in table_schema.primary_keys
            ]
        if keys:
            sample_data = sample_data.drop_duplicates(subset=keys)
            return sample_data.reset_index(drop=True)
        else:
            return sample_data

    @log_time_factory(logger)
    @synthesizer_sample
    def sample(
        self,
        n_samples: float | None = 1.0,
        connector: RDBMSConnector | None = None,
        if_exists: str | WriteMode = WriteMode.APPEND,
        random_state: RandomSeed = None
    ) -> MultiDataset:
        """Sample from a trained multitable synthesizer.

        Args:
            n_samples (float | None): percentage of the original database to sample. Values between 0.1 up to 5 are accepted by the method. Default is set to 1.0.
            connector (RDBMSConnector | None): connector to enable persist tables progressively.
            if_exists ({'fail', 'replace', 'append'}): defines the write behavior when the table already exists. Defaults to 'append'
                - **append:** add the data to the pre-existing table.
                - **fail:** raises an error if the table exists.
                - **replace:** drop the existing table and create a new one.
                Note that when using replace, if the database table has constraints that restrict deletion,
                the persistence can fail leading to inconsistencies in the database
            random_state: random generator or seed for the synthesizer fit

        Note:
            When this method receives a connector to a RDBMS database, it persists all generated tables and return an empty dataset. The use of a connector is recommended for
            a better memory management.

        Returns:
            synthetic MultiDataset
        """
        super().sample(n_samples=n_samples)

        _n_samples = n_samples #TODO check if this can be removed
        composite_keys = self._get_user_defined_composite_keys()

        shapes = {
            t: get_expected_size(nrows, _n_samples)
            for t, nrows in self.tables_nrows.items()
        }

        if_exists = WriteMode(if_exists)
        if connector is not None and if_exists == WriteMode.REPLACE:
            logger.info(
                "[SYNTHESIZER] - Replace selected, deleting previous records from synthetic database.")
            # Deletes the rows instead of the tables to preserve the constraints
            connector.delete_tables(self._topological_order[::-1])
            if_exists = WriteMode.APPEND

        sample_tables = {}
        persisted_tables = set()
        for i, (table, previous_table) in enumerate(self._tables_order):
            logger.info(
                f"({i+1}/{len(self._tables_order)}) - Synthesizing table: {table}")
            logger.debug(
                f"Conditioned on: {previous_table if previous_table else 'None'}")

            synthesizer_data = self._synthesizers[table]
            model = synthesizer_data["model"]
            table_schema = self.schema.get(table, TableSchema())
            if isinstance(model, TimeSeriesSynthesizer):
                raise NotImplementedError

            if self.is_attribute_table(table):
                assert isinstance(
                    model, FakerSynthesizer
                ), f"Model for attribute table {table} was incorrectly defined."
                sample_data = model._MultiTableSynthesizer__sample(model.nrows)
                sample_data = {table: sample_data.to_pandas()}
            elif not previous_table:
                sample_data = self._sample_pipeline(
                    synthesizer=model, n_samples=shapes[table], table=table, random_state=random_state)

                sample_data = self._postprocess_table(
                    synthesizer=model, tbl_schema=table_schema, name=table, data=sample_data, card=shapes[table])
                sample_data[table] = self.anonymize_table_pks(
                    table=table, table_df=sample_data[table])
            elif len(previous_table) == 1:
                previous_table = previous_table[0]
                condition_df = self.get_condition_df(
                    table, previous_table, sample_tables, shapes, _n_samples)
                logger.debug(
                    f"p({table}|{previous_table}) condition df size: {condition_df.shape}")

                shapes[table] = condition_df.shape[0]

                # sample pipeline
                condition_dataset = model.dataset_preprocessor.transform(
                    Dataset(condition_df)).to_pandas()

                sample_data = self._sample_pipeline(
                    synthesizer=model, n_samples=shapes[table], table=table, condition_df=condition_dataset, random_state=random_state)
                sample_data = {table: sample_data}

                table_key, previous_table_key = self._get_related_keys(
                    table, previous_table)

                fk_column = f"{previous_table}.{previous_table_key}" \
                    if isinstance(previous_table_key, str) \
                    else f"{previous_table}.{previous_table_key[0]}"

                fk_values = sample_data[table][fk_column]
                sample_data[table] = sample_data[table][[
                    c for c in sample_data[table].columns if c.startswith(f"{table}.")]]

                sample_data = self._postprocess_table(
                    synthesizer=model, tbl_schema=table_schema, name=table, data=sample_data[table], card=shapes[table])

                # update previous table data references
                self._update_key_references(
                    table, previous_table, sample_data, condition_df, fk_values, sample_tables)

                sample_data[table] = self.anonymize_table_pks(
                    table, sample_data[table])
            else:
                oversample_factor = 1.1
                sample_data = self._sample_pipeline(
                    synthesizer=model,
                    n_samples=int(shapes[table] * oversample_factor),
                    table=table,
                    random_state=random_state
                )
                for parent in previous_table:
                    table_key, parent_key = self._get_related_keys(
                        table, parent)
                    # parent has only one fk
                    if isinstance(parent_key, str):
                        pool = sample_tables[parent][[parent_key] + self._encoded_columns[parent]].groupby(
                            self._encoded_columns[parent]).agg(list).to_dict()[parent_key]
                        unseen_encodings = set()
                        # TODO extract to a decoding method
                        # currently only support encoding multiple columns
                        encoding = f"{parent}.{self._encoded_columns[parent][0]}"
                        reference = f"{table}.{table_key}"
                        unique_encodings = sample_data[encoding].unique()
                        for code in unique_encodings:
                            if code not in pool:
                                unseen_encodings.add(code)
                                continue
                            mask = (sample_data[encoding] == code)
                            replace = mask.sum() > len(pool[code])

                            parent_dtype = sample_tables[parent][parent_key].dtype
                            if (
                                sample_data[reference].isna().any()
                                and is_integer_dtype(parent_dtype)
                            ):
                                parent_dtype = float
                            sample_data[reference] = sample_data[reference].astype(
                                parent_dtype
                            )
                            sample_data.loc[mask, reference] = choice(
                                pool[code], replace=replace, size=mask.sum())
                        if unseen_encodings:
                            mask = sample_data[encoding].isin(unseen_encodings)
                            sample_data.loc[mask, reference] = choice(
                                sample_tables[parent][parent_key].unique(),
                                replace=True,
                                size=mask.sum()
                            )
                    # table key is a composite key
                    else:
                        base_table = sample_tables[parent][parent_key +
                                                           self._encoded_columns[parent]]
                        base_table["__key__"] = base_table[parent_key].apply(
                            lambda row: "|".join(row.values.astype(str)), axis=1)
                        pool = base_table[["__key__"] + self._encoded_columns[parent]].groupby(
                            self._encoded_columns[parent]).agg(list).to_dict()["__key__"]
                        unseen_encodings = set()
                        encoding = f"{parent}.{self._encoded_columns[parent][0]}"
                        unique_encodings = sample_data[encoding].dropna(
                        ).unique()
                        references = []
                        # temporarily convert types to string
                        for t_key in table_key:
                            reference = f"{table}.{t_key}"
                            references.append(reference)
                            sample_data[reference] = sample_data[reference].astype(
                                str)

                        for code in unique_encodings:
                            if code not in pool:
                                unseen_encodings.add(code)
                                continue

                            mask = (sample_data[encoding] == code)
                            replace = mask.sum() > len(pool[code])
                            selected_codes = choice(
                                pool[code], replace=replace, size=mask.sum())
                            sample_data.loc[mask, references] = [
                                x.split("|") for x in selected_codes]

                        if unseen_encodings:
                            mask = sample_data[encoding].isin(unseen_encodings)
                            merged_pool = set().union(
                                *[set(v) for v in pool.values()])
                            selected_codes = choice(
                                merged_pool, replace=True, size=mask.sum())
                            sample_data.loc[mask, references] = [
                                x.split("|") for x in selected_codes]

                        # revert columns types to the original values
                        for t_key, p_key in zip(table_key, parent_key):
                            reference = f"{table}.{t_key}"
                            parent_dtype = sample_tables[parent][p_key].dtype
                            if (
                                sample_data[reference].isna().any()
                                and is_integer_dtype(parent_dtype)
                            ):
                                parent_dtype = float
                            sample_data[reference] = sample_data[reference].astype(
                                parent_dtype
                            )

                # Drop duplicated keys
                sample_data = self._remove_duplicated_primary_keys(
                    sample_data, table_schema, table)
                # removes the effect of oversampling factor
                if len(sample_data) > shapes[table]:
                    sample_data = sample_data.head(shapes[table])
                sample_data = sample_data[[
                    c for c in sample_data.columns
                    if c.startswith(f"{table}.")
                ]]
                sample_data = self._postprocess_table(
                    synthesizer=model, tbl_schema=table_schema, name=table, data=sample_data, card=shapes[table])
                sample_data[table] = self.anonymize_table_pks(
                    table=table, table_df=sample_data[table])

            if len(table_schema.primary_keys) > 1:
                sample_data[table] = self._remove_duplicated_primary_keys(
                    sample_data[table], table_schema, table)
            sample_tables.update(sample_data)

            if connector:
                models = {k: v["model"] for k, v in self._synthesizers.items()}
                persisted_tables = self._persist_unrequired_tables(
                    synthesizer=models,
                    tables=list(self._synthesizers.keys()),
                    sample_tables=sample_tables,
                    persisted_tables=persisted_tables,
                    connector=connector,
                    if_exists=if_exists,
                    cardinality=shapes,
                    is_postprocessed=True,
                    composite_keys=composite_keys,
                )
                logger.debug(f"Persisted tables: {persisted_tables}")

        # insert empty tables and constant here
        for t, v in self.empty_tables.items():
            sample_tables[t] = v.copy()

        for t, v in self.constant_tables.items():
            sample_tables[t] = v._data.copy()

        # postprocess tables
        for table, table_df in sample_tables.items():
            table_schema = self.schema.get(table, TableSchema())

            table_df = replace_for_valid_keys(
                table_schema=table_schema,
                sample_data=table_df,
                sample_tables=sample_tables,
                missing_values=self.missing_values[table],
                reference_values=self._key_values,
            )

        self._handle_user_defined_composite_keys(composite_keys, sample_tables)

        for table, table_df in sample_tables.items():
            table_schema = self.schema.get(table, TableSchema())
            table_df = apply_table_calculated_features(
                calculated_features=self.calculated_features,
                table=table,
                table_data=table_df,
                sample_tables=sample_tables,
            )
            if table in self._encoded_columns:
                table_df = table_df.drop(
                    columns=self._encoded_columns[table])
            sample_tables[table] = table_df

        if connector:
            # persist remaining tables following the topological order
            tables_to_persist = [
                table for table in self._topological_order
                if table not in persisted_tables
            ]
            for table in tables_to_persist:
                self._persist_table(
                    table=table,
                    sample_tables=sample_tables,
                    connector=connector,
                    if_exists=if_exists,
                )
            sample_tables = {}
        else:
            sample_tables = {k: Dataset(v) for k, v in sample_tables.items()}

        self._clear_reference_values()
        return MultiDataset(datasets=sample_tables, schema=self.schema)

    def _compute_cardinality_distribution(self, X: MultiDataset, metadata: MultiMetadata):
        def _add_cardinality(missing_foreign_card, metadata, table, column):
            card = metadata[table].summary['value_counts'].get(column)
            if card is None:
                if table not in missing_foreign_card:
                    missing_foreign_card[table] = []
                    missing_foreign_card[table].append(column)
            else:
                self.cardinality_dist[table][column] = pdDataFrame(
                    card.values)[0].value_counts()

        missing_foreign_card = {}
        for t, rels in self.schema.items():
            if t in self.empty_tables or t in self.constant_tables:
                continue
            if t not in self.cardinality_dist:
                self.cardinality_dist[t] = {}
            for fk in rels.foreign_keys:
                _add_cardinality(missing_foreign_card, metadata, t, fk.column)
            for pk in rels.primary_keys:
                if pk not in self.cardinality_dist[t]:
                    _add_cardinality(missing_foreign_card, metadata, t, pk)

        # Compute the value counts when it is not available already in the Metadata object
        for t, missing_cols in missing_foreign_card.items():
            no_null = {col: drop_null(
                X[t]._data[col], is_str=metadata[t].columns[col].vartype == VariableType.STR) for col in missing_cols}
            add_value_counts = {
                col: no_null[col].value_counts() for col in missing_cols}
            # FIXME This seems wrong since add cardinality uses value_counts().value_counts() and here only uses value_counts()
            self.cardinality_dist[t].update(compute(add_value_counts)[0])

    def _drop_null_rows(
        self,
        df: pdDataFrame,
        relationship: TableSchema,
        table_name: str
    ) -> pdDataFrame:
        """Remove table's empty rows. A row is considered empty if all columns
        except the primary and foreign keys are empty.

        Args:
            df (pdDataFrame): table data to analyse
            relationship (TableSchema): table relationships schema
            table_name (str): table name.

        Returns:
            pdDataFrame: table with empty rows removed
        """
        if "." in df.columns[0]:
            fks = [f"{fk.table}.{fk.column}" for fk in relationship.foreign_keys]
            pks = [f"{table_name}.{pk}" for pk in relationship.primary_keys]
        else:
            fks = [fk.column for fk in relationship.foreign_keys]
            pks = relationship.primary_keys

        keys = pks + fks
        non_keys = [k for k in df.columns if k not in keys]
        mask = df[non_keys].dropna(how="all").index
        return df.loc[mask].reset_index(drop=True)

    def _update_unnatend_fks_encodings(
        self,
        sample_data: dict[str, pdDataFrame],
        table: str,
        previous_table: str
    ):
        table_schema = self.schema.get(table, TableSchema())
        for fk in table_schema.foreign_keys:
            if fk.parent_table != previous_table:
                has_missing = self.missing_values[table].get(fk.column, 0) > 0
                dtype = str(sample_data[table][fk.column].dtype)
                if has_missing and dtype.startswith('float'):
                    missing_mask = sample_data[table][fk.column] == -1
                    sample_data[table][fk.column].loc[missing_mask] = npnan
                elif has_missing and dtype.startswith('int'):
                    sample_data[table][fk.column] = sample_data[table][fk.column].astype(
                        'float')
                    missing_mask = sample_data[table][fk.column] == -1
                    sample_data[table][fk.column].loc[missing_mask] = npnan
        return sample_data

    def _update_key_references(self, table, previous_table, sample_data, condition_df, fk_values, sample_tables):
        def get_table_column(df: pdDataFrame, table: str, column: str | list[str]) -> str | list[str]:
            columns = [column] if isinstance(column, str) else column
            columns = [
                f"{table}.{col}" if f"{table}.{col}" in df.columns else col
                for col in columns
            ]
            if len(columns) == 1:
                return columns[0]
            return columns

        # get keys
        table_key, previous_table_key = self._get_related_keys(
            table, previous_table)
        previous_table_col = get_table_column(
            condition_df, previous_table, previous_table_key)
        table_col = get_table_column(
            sample_data[table], table, table_key)

        if isinstance(table_key, list):
            table_key = table_key[0]
            previous_table_key = previous_table_key[0]

        # update base fk
        sample_data[table][table_col] = condition_df[previous_table_col]

        # fix other fk missing encodings
        self._update_unnatend_fks_encodings(
            sample_data=sample_data,
            table=table,
            previous_table=previous_table
        )

    def _get_related_keys(self, table: str, parent_table: str) -> Tuple[str | list[str], str | list[str]] | None:
        def get_column_names(columns: str | list[str]) -> str | list[str]:
            if isinstance(columns, str):
                return columns.split(".", 1)[-1]
            return [col.split(".", 1)[-1] for col in columns]

        for child, parent, keys in self.relationships:
            # table is the child table
            if parent == parent_table and child == table:
                return get_column_names(keys["right_on"]), get_column_names(keys["left_on"])
            if child == parent_table and parent == table:
                return get_column_names(keys["left_on"]), get_column_names(keys["right_on"])

        # tables are not related
        return None

    def _persist_table(
        self,
        table: str,
        sample_tables: dict,
        connector: RDBMSConnector,
        if_exists: str | WriteMode
    ):
        logger.info(
            f"Persisting {len(sample_tables[table])} rows in table [{table}]")

        def get_col_name(name: str) -> str:
            if "." in name:
                return name.split(".", 1)[-1]
            return name

        table_data = sample_tables[table].rename(
            get_col_name,
            axis="columns"
        )
        connector.write_table(Dataset(table_data),
                              schema_name=connector.schema_name,
                              name=table, if_exists=if_exists)

    def _persist_unrequired_tables(
        self,
        synthesizer: BaseModel | dict[str, BaseModel],
        tables: list[str],
        sample_tables: dict,
        persisted_tables: set,
        connector: RDBMSConnector,
        cardinality: dict[str, int],
        if_exists: str | WriteMode = WriteMode.APPEND,
        is_postprocessed: bool = False,
        composite_keys: dict = None,
    ) -> list:
        """Persists tables that are not necessary for synth anymore.

        Note:
            alters samples inplace.

        Args:
            synthesizer (BaseModel): synthesizer being sampled.
            tables (list[str]): tables being sampled.
            sample_tables (dict): collection of sampled tables.
            persisted_tables (set): list of tables already persisted
            connector (RDBMSConnector): database connector
            cardinatity (dict[str, int]): expected table sample cardinality
            is_postprocessed (bool): indicates if the sampled tables are already post processed. Defaults to false

        Returns:
            list: updated list of persisted tables
        """
        composite_keys = composite_keys or {}
        tables_to_persist = get_tables_to_persist(
            tables=tables,
            sampled_tables=sample_tables,
            persisted_tables=persisted_tables,
            schema=self.schema,
            composite_keys=composite_keys,
            calculated_features=self.calculated_features
        )

        logger.debug(f"Tables to persist: [{tables_to_persist}]")
        processed_tables = {}
        for table in self._topological_order:
            if table not in tables_to_persist:
                continue

            # check if the parent tables are already persisted
            table_schema = self.schema.get(table, TableSchema())
            if not is_parent_persisted(table_schema, persisted_tables):
                # persisting child tables may without the parent may fail because of pk-fks constraints
                logger.debug(
                    f"Unmet dependencies for table [{table}]. Table was not persisted.")
                continue

            # post process table
            if not is_postprocessed:
                model = synthesizer if isinstance(
                    synthesizer, BaseModel) else synthesizer[table]
                table_data = self._postprocess_table(
                    model,
                    table_schema,
                    table,
                    sample_tables[table],
                    cardinality[table]
                )
            else:
                table_data = {table: sample_tables[table]}

            # Handles the invalid relationships from childs with multiple parents and diamond relationships
            table_data[table] = replace_for_valid_keys(
                table_schema=table_schema,
                sample_data=table_data[table],
                sample_tables=sample_tables,
                missing_values=self.missing_values[table],
                reference_values=self._key_values,
            )
            self.__update_valid_fk_values(table, table_data, persisted_tables)

            table_data[table] = apply_table_calculated_features(
                calculated_features=self.calculated_features,
                table=table,
                table_data=table_data[table],
                sample_tables=sample_tables,
            )
            # Drop encoded columns
            if table in self._encoded_columns:
                table_data[table] = table_data[table].drop(
                    columns=self._encoded_columns[table])

            processed_tables.update(table_data)
            sample_tables[table] = table_data[table]

        self._handle_user_defined_composite_keys(composite_keys, sample_tables)

        for table in self._topological_order:
            if table not in tables_to_persist:
                continue
            # persist table
            self._persist_table(table, processed_tables, connector, if_exists)
            persisted_tables.add(table)

            # free table memory
            sample_tables.pop(table)
            processed_tables.pop(table)

        return persisted_tables

    def __update_valid_fk_values(self, table, table_data, persisted_tables):
        if table not in self._key_values:
            self._key_values[table] = {}

        # check if the table is referenced by other tables
        references = self.references.get(table, [])
        is_referenced = any([
            t for t in
            [fk.parent_table for fk in references]
            if t not in persisted_tables
        ])
        # if there is no more references clear the values
        if not is_referenced:
            self._key_values.pop(table)
            return
        # if there are references, store valid FKs
        for fk in references:
            self._key_values[table][fk.parent_column] = table_data[table][fk.parent_column].unique(
            )

    def _clear_reference_values(self):
        """Clear sampled FK's reference values."""
        self._key_values = {}

    @log_time_factory(logger)
    def _postprocess_table(self, synthesizer: BaseModel, tbl_schema: TableSchema, name: str, data: pdDataFrame, card: int) -> dict:
        table_data = {name: data}

        fks = [f"{name}.{fk.column}" for fk in tbl_schema.foreign_keys]
        pks = [
            f"{name}.{pk}" for pk in tbl_schema.primary_keys if f"{name}.{pk}" not in fks]
        table_keys = pks + fks
        non_key_cols = [
            col for col in table_data[name].columns if col not in table_keys]

        # Constant columns must not be post-processed.
        non_key_cols = [
            c for c in non_key_cols if c not in synthesizer._constant_features.keys()]

        # PKs and FKs are not inverse transformed.
        if non_key_cols:
            table_data[name] = synthesizer._pandas_df_to_dask(
                table_data[name])
            table_data[name][non_key_cols] = synthesizer.dataset_preprocessor.inverse_transform(
                table_data[name][non_key_cols].copy())
            table_data[name] = table_data[name].compute()

            table_data = self._postprocess_missing_values(table_data)
            table_data = self._postprocess_NA_values(
                table_data, self.tables_schemas)

            table_data[name] = self._postprocess_NA_categoricals(
                table_data[name])
            table_data[name] = self._drop_null_rows(
                df=table_data[name],
                relationship=self.schema.get(name, TableSchema()),
                table_name=name,
            )[:card]
        else:
            table_data[name] = table_data[name][:card]

        table_data[name] = table_data[name].rename(
            columns={c: c.split(".", 1)[-1] for c in table_data[name].columns})
        return table_data

    def _sample_data(
        self,
        synthesizer: BaseModel,
        n_samples: float | int | None = None,
        condition_on: list[ConditionalFeature] | dict | None = None,
        n_entities: int | None = None,
        random_state: RandomSeed = None
    ) -> pdDataFrame:
        """Bind the argument to the proper synthesizer.

        Args:
            synthesizer (BaseModel): synthesizer to sample
            n_samples (float | int | None): proportion or number of samples
            condition_on (list[ConditionalFeature] | dict | None): conditional features
            n_entities (int | None): number of entities

        Returns:
            pdDataFrame
        """
        if isinstance(synthesizer, TimeSeriesSynthesizer):
            result = synthesizer._MultiTableSynthesizer__sample(
                n_entities=n_entities, sort_result=False, condition_on=condition_on, random_state=random_state)
        else:
            result = synthesizer._MultiTableSynthesizer__sample(
                n_samples, condition_on=condition_on, random_state=random_state)

        return result.to_pandas()

    @log_time_factory(logger)
    def _sample_pipeline(
        self,
        synthesizer: BaseModel,
        n_samples: int,
        table: str,
        condition_df: pdDataFrame | None = None,
        random_state: RandomSeed = None
    ):
        sample_data = self._sample_data(
            synthesizer, n_samples=n_samples, condition_on=condition_df, random_state=random_state)
        sample_data = self._replace_pks(table, sample_data)
        return sample_data

    def _replace_pks(self, t_name: str, t_df: pdDataFrame) -> pdDataFrame:
        fks = [fk.column for fk in self.schema[t_name].foreign_keys]
        for pk in self.schema[t_name].primary_keys:
            if pk not in fks:
                t_df.loc[:, f"{t_name}.{pk}"] = range(1, t_df.shape[0] + 1)
        return t_df

    def save(self, path: str):
        """Saves the SYNTHESIZER and the models fitted per variable."""

        logger.info("[SYNTHESIZER] - Saving SYNTHESIZER state.")

        with open(path, "wb") as f:
            pdump(self, f)
            f.close()

    @classmethod
    def load(cls, path: str):
        logger.info("[SYNTHESIZER] - Loading SYNTHESIZER state.")
        with open(path, "rb") as f:
            synth = pload(f)

        assert isinstance(synth, MultiTableSynthesizer), (
            "The loaded file must correspond to a MultiTableSynthesizer object. "
            "Please validate the given input path."
        )

        return synth
