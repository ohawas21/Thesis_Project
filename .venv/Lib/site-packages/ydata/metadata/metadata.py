"""Generic implementation of Metadata class that contains data about a Dataset
class."""
from __future__ import annotations

import warnings
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from os import getenv
from pickle import HIGHEST_PROTOCOL, dump, load
from typing import Iterable, List, Union, Optional
from warnings import warn

from dask import compute
from dask.array import isinf
from dask.array.stats import skew
from dask.diagnostics import ProgressBar
from dask.distributed import Future
from numpy import ceil as npceil
from numpy import log2
from pandas import DataFrame as pdDataframe
from pandas import Series as pdSeries
from scipy.stats import entropy

from ydata.characteristics import ColumnCharacteristic
from ydata.characteristics.presidio import detect_characteristics as presidio_detect_characteristics
from ydata.dataset import Dataset, DatasetType
from ydata.metadata.builder import MetadataConfigurationBuilder
from ydata.metadata.column import Column
from ydata.metadata.compute import ComputeConfig, ComputeMode
from ydata.metadata.utils import drop_null
from ydata.metadata.warning_types import ConstantLengthWarning  # pylint: disable = W0622
from ydata.metadata.warning_types import (ConstantWarning, CorrelationWarning, DuplicatesWarning,
                                          HighCardinalityWarning, ImbalanceWarning, InfinityWarning,
                                          MissingValuesWarning, SkewnessWarning, UniqueWarning, WarningEngine,
                                          WarningType, ZerosWarning)
from ydata.utils.associations import association_matrix
from ydata.utils.configuration import TextStyle
from ydata.utils.dask import DaskCluster
from ydata.utils.data_types import _NUMERICAL_VARTYPES, CATEGORICAL_DTYPES, DATA_VARTYPE_MAP, DataType, VariableType
from ydata.utils.exceptions import (DataTypeRequestError, InvalidDatasetTypeError,
                                    InvalidEntityColumnError, InvalidMetadataInputError)
from ydata.utils.logger import utilslogger_config
from ydata.utils.misc import log_time_factory
from ydata.utils.sampling.proportion import SAMPLE_SIZES, calculate_wilson_cc, determine_z_value

logger = utilslogger_config(verbose=getenv(
    "VERBOSE", "false").lower() == "true")

DEFAULT_PARTITION_SIZE: int = 300


def assign_correlation(data: Future | pdDataframe, m: Metadata):
    if isinstance(data, pdDataframe) or data.status == 'finished':
        m.summary['correlation'] = data if isinstance(
            data, pdDataframe) else data.result()
        m._Metadata__warnings = m._get_warnings()


def assign_characteristics(data: Future | tuple, m: Metadata, schema, columns: dict | None = None):
    if isinstance(data, tuple) or data.status == 'finished':
        data = data.result() if isinstance(data, Future) else data
        characteristics, tags = data[0], data[1]
        m.summary['characteristics'] = characteristics
        m._Metadata__columns = m._get_columns_metadata(
            word_count=m.summary["word_count"],
            cardinality=m.summary['cardinality'],
            characteristics=tags,
            schema=schema,
            columns=columns
        )


@dataclass
class DatasetAttr:
    sortbykey: List[str] = field(default_factory=lambda: ['colA'])
    entities: List[str] = field(default_factory=list)

    def __init__(self, sortbykey: Union[List[str], str], entities: Optional[Union[List[str], str]] = None):
        self.sortbykey = sortbykey if isinstance(sortbykey, list) else [sortbykey]
        self.entities = entities if isinstance(entities, list) else [entities] if entities is not None else []

    @staticmethod
    def fields():
        return ["sortbykey", "entities"]

    def empty(self):
        return not any(getattr(self, c) for c in self.fields())


# pylint: disable = W0212
def istype(d: dict, inputtype=str):
    """Validate wether all the values from a dict are of a provided type."""
    return all(isinstance(val, inputtype) for val in d.values())


# @typechecked
def valid_input_col(d: dict):
    """Validate input column."""
    is_str = istype(d, str)
    is_datatype = istype(d, DataType)

    assert (
        is_str or is_datatype
    ), "Please provide a valid dictionary with the Data types definition for the dataset columns."

    if is_str:
        valid_dtypes = [d.value for d in DataType]
        assert all(
            val in valid_dtypes for val in d.values()
        ), f"Please provide valid Data Types for the columns. Valid data types are: {valid_dtypes}"


# @typechecked
class Metadata:
    """
    Core metadata class for analyzing Datasets.

    The `Metadata` class is responsible for **extracting statistical summaries** and
    **data characteristics** from a given `Dataset`. It plays a central role in both
    **data profiling** and **synthetic data generation**, providing insights into
    feature distributions, unique values, correlations, and other dataset properties.

    ## Key Features:
    - **Schema Inference**: Identifies feature types (`DataTypes`) based on data characteristics.
    - **Descriptive Statistics**: Computes **uniques, skewness, correlations, distributions, among other metrics**.
    - **Profiling Support**: Helps analyze dataset structure, feature importance, and warnings.
    - **Synthetic Data Generation Support**: Assists in learning data characteristics and identification of potential PII data.
    - **Configurable Computation**: Supports partitioning and configurable metrics for large datasets.

    Properties:
        columns (List[str]): List of feature names in the dataset.
        ncols (int): number of features/columns
        shape (Tuple[int, int]): tuple of (nrows, ncols)
        uniques (Dict[str, int]): number of unique values per feature.
        skewness (Dict[str, float]): skewness metric per continuous feature.
        schema (Dict[str, str]): feature type (VariableTypes), based on data types.

        ## Example Usage:
        ```python
        from ydata.metadata import Dataset, Metadata

        # Create a dataset
        df = pd.read_csv('data.csv')
        dataset = Dataset(df)

        # Generate metadata for Dataset analysis
        metadata = Metadata(dataset=dataset)

        # Access dataset insights
        print(metadata.shape)      # (10000, 12)
        print(metadata.schema)     # {'age': 'int', 'salary': 'float', 'category': 'string'}
        print(metadata.uniques)    # {'age': 50, 'salary': 2000, 'category': 5}
        ```
    """

    DATASET_WARNINGS: list[str] = ["duplicates"]
    MIN_ROWS_FOR_SAMPLING: int = 1_000_000
    MAX_CORR_CARDINALITY: int = 100

    def __init__(
        self,
        dataset: Dataset | None = None,
        dataset_attrs: dict | None = None,
        columns: dict | None = None,
        dataset_type: DatasetType | str = DatasetType.TABULAR,
        infer_characteristics: bool = False,
        characteristics: dict | None = None,
        pairwise_metrics: bool = True,
        partition_size: int = DEFAULT_PARTITION_SIZE,
        intercolumns_warnings: bool = True,
        compute_config: ComputeConfig | dict | None = None,
        configuration_builder: MetadataConfigurationBuilder | None = None,
        partition: bool = False
    ):
        """
        Initialize a `Metadata` object for dataset profiling and synthetic data generation.

        The `Metadata` class extracts statistical insights and data characteristics from a dataset.
        It supports **automatic inference of feature types, pairwise relationships, and
        data distribution metrics**, making it a core component for **data profiling** and
        **synthetic data modeling**.

        Args:
            dataset (Dataset | None): The dataset to analyze. If `None`, metadata can be constructed using the configuration_builder to generate Fake synthetic datasets.
            dataset_attrs (dict | None): Dictionary of metadata attributes. This is particularly important to be defined for time-series datasets.
            columns (List[str]): List of feature names in the dataset to be included for the calculated Metadata.
            dataset_type (`DatasetType | str`, default=`DatasetType.TABULAR`):   The type of dataset being analyzed. Valid values are `tabular` or `time-series`. Defaults to `tabular`.
            infer_characteristics (bool, default=`False`): Whether to automatically infer data characteristics such as potential PII information (email, name, etc.). Defaults to False.
            characteristics (`dict | None`, default=`None`):  A dictionary specifying already known characteristics/PII information. Must be provided in the following format {column_name: pii_type}.
            pairwise_metrics (`bool`, default=`True`):  Whether to compute **pairwise correlation metrics** for the Dataset. Defaults to True. For dataset with a big number of columns it is recommended to set it to False.
            partition_size (`int`, default=300): Defines the partition size for distributed computation, based on the number of columns, improving efficiency on large datasets.
            intercolumns_warnings (`bool`, default=`True`): If `True`, generates warnings related to inter-column relationships, indicating whether two columns are the same or equal.
            configuration_builder (`MetadataConfigurationBuilder | None`, default=`None`): A builder for defining a structure of a Dataset as well as its variable, data types and statistical properties.
                                                                                           It is only considered if dataset is None.
        """
        self.__columns = None
        self._target = None
        self.__warnings = None
        self._cols_dtypes = None
        self._metadata = {}
        self._manual_characteristics = characteristics if characteristics is not None else {}
        self._characteristics_confidence_level = 0.99  # 1% error
        self._characteristics_threshold = 0.8
        self._is_multitable: bool = False
        self._categorical_threshold = 30
        self._wordcount_threshold = 5
        self._partition_size = partition_size
        self.__intercolumns_warnings = intercolumns_warnings
        self._dataset_attrs = None
        # Indicate if the base computation is done.
        self._base_computed: bool = False
        self.__partition: bool = partition
        self.dataset_type = dataset_type

        if dataset is not None and configuration_builder:
            raise ValueError(
                "Dataset and MetadataConfigurationBuilder are excludents. Please pass one or the other."
            )

        self._compute_config: ComputeConfig = self.__init_compute_config(
            config=compute_config,
            pairwise_metrics=pairwise_metrics,
            infer_characteristics=infer_characteristics
        )

        self.status: dict = {}

        if dataset is not None:
            if columns:
                dataset = dataset[list(columns.keys())]

            self.__call__(dataset, dataset_attrs, columns, dataset_type)

        elif configuration_builder is not None:
            self.__build_metadata_from_config(configuration_builder)

    def __init_compute_config(self, config: ComputeConfig | dict | None, pairwise_metrics: bool, infer_characteristics: bool) -> ComputeConfig:
        config = config if config is not None else {}
        config = config if isinstance(
            config, ComputeConfig) else ComputeConfig(**config)
        config.infer_characteristics = infer_characteristics
        config.pairwise_metrics = pairwise_metrics
        return config

    def __validate_dataset_attrs(self, dataset_attrs, columns: list | None = None):
        cols = columns if columns is not None else list(self.__columns.keys())

        try:
            sortbykey = dataset_attrs.sortbykey
            if not all(e in list(cols) for e in sortbykey):
                raise InvalidMetadataInputError(
                    "The specified **sortbykey** must be a column/list of columns that exists in the provided dataset. "
                    "Please validate your inputs."
                )

            if len(dataset_attrs.entities)>0:
                if not all(e in list(cols) for e in dataset_attrs.entities):
                    raise InvalidMetadataInputError(
                        "The specified **entities** must be a column/list of columns that exists in the provided dataset. "
                        "Please validate your inputs."
                    )

        except TypeError as e:
            dataset_attrs_fields = ", ".join(DatasetAttr.fields())
            raise InvalidMetadataInputError(
                f"Please provide valid dataset attributes. {e} is not a valid key."
                f"{dataset_attrs_fields} are expected."
            )

    def __valid_inputs(self, dataset, columns):
        if columns is not None:
            valid_input_col(d=columns)
            columns_exist = all(
                col in dataset.columns for col in columns.keys())
            assert (
                columns_exist
            ), "Some of the keys provided for the columns do not match the provided dataset"

            schema = dataset.schema
            for col, datatype in columns.items():
                dtype = DataType(datatype)
                compatible_vartypes = DATA_VARTYPE_MAP[dtype]
                vtype = VariableType(schema[col])
                if vtype not in compatible_vartypes:
                    str_compatible_vartypes = ', '.join(
                        [v.value for v in compatible_vartypes])
                    raise DataTypeRequestError(
                        f"The requested datatype '{dtype.value}' for column '{col}' is not compatible with its variable type '{vtype.value}'.\nCompatible datatypes are: {str_compatible_vartypes}")

    @log_time_factory(logger)
    def _init_metadata(
        self,
        dataset: Dataset,
        columns: dict | None = None,
    ):
        self._compute_config.resolve_auto(inplace=True)
        logger.debug("[METADATA] - Validating the dataset inputs.")
        self.__valid_inputs(dataset=dataset, columns=columns)
        logger.debug("[METADATA] - Obtaining the metadata for the dataset.")
        self._compute_metadata(dataset, columns=columns)
        self._deduce_characteristics()
        return self

    def __create_partition(
        self,
        dataset: Dataset,
        cols: List[str],
        partition_columns: dict,
    ) -> Metadata:
        meta = Metadata(
            dataset=dataset[cols],
            dataset_attrs=None,
            columns=partition_columns,
            dataset_type=self.dataset_type,
            infer_characteristics=self._compute_config.infer_characteristics,
            characteristics=self._manual_characteristics,
            pairwise_metrics=False,
            intercolumns_warnings=False,  # can I remove this and use only parwise_metrics?
            partition_size=self._partition_size,
            partition=True,
        )
        meta._init_metadata(
            dataset[cols], partition_columns)
        return meta

    def __validate_dataset_columns(self, dataset: Dataset):
        if len(dataset.columns) > len(set(dataset.columns)):
            raise InvalidMetadataInputError(
                "Your dataset has 2 columns with the same name. Please revisit your inputs before moving forward.")

    @log_time_factory(logger)
    def __call__(
        self,
        dataset: Dataset,
        dataset_attrs: dict | DatasetAttr | None = None,
        columns: dict | None = None,
        dataset_type: DatasetType | str = DatasetType.TABULAR,
    ) -> "Metadata":

        self._base_computed = True

        #validates duplicate column name
        self.__validate_dataset_columns(dataset)

        #validates and sets Metadata dataset type
        self.__set_dataset_type(DatasetType(dataset_type), dataset_attrs)

        if len(dataset.columns) > self._partition_size:
            # disable pairwise metrics
            self._compute_config.pairwise_metrics = False
            self.__intercolumns_warnings = False
            n_partitions = int(
                npceil(len(dataset.columns) / self._partition_size))
            columns = columns if columns else {}
            logger.info(
                f"Dataset has {len(dataset.columns)} columns. "
                f"Spliting in columns in {n_partitions} partitions with at most {self._partition_size} cols."
            )
            for i in range(n_partitions):
                logger.info(
                    f"Computing Metadata partition {i}/{n_partitions}")

                pivot = i * self._partition_size
                cols = dataset.columns[pivot:pivot+self._partition_size]
                partition_columns = {k: v for k,
                                     v in columns.items() if k in cols}
                if i == 0:
                    self._init_metadata(
                        dataset[cols], partition_columns)
                else:
                    meta = self.__create_partition(
                        dataset, cols, partition_columns)
                    self.combine(meta)
        else:
            self._init_metadata(dataset, columns)

        # validates and sets Metadata dataset attributes
        if dataset_attrs:
            dataset_attrs = DatasetAttr(**dataset_attrs)
            self.__validate_dataset_attrs(dataset_attrs, columns=dataset.columns)
            self._dataset_attrs = dataset_attrs

        return self

    def __set_dataset_type(self, dataset_type,
                                 dataset_attrs: dict | None = None):
        try:
            if dataset_attrs is None:
                dataset_attrs = self._dataset_attrs
            elif isinstance(dataset_attrs, dict):
                dataset_attrs = DatasetAttr(**dataset_attrs)

            self.dataset_type = DatasetType(dataset_type)
        except ValueError:
            raise InvalidDatasetTypeError(
                f"Provided dataset_type {dataset_type} is not valid."
            )

    def set_dataset_type(
        self,
        dataset_type: DatasetType | str,
        dataset_attrs: dict | None = None,
    ):
        """
        Update the dataset type and optionally set dataset attributes.

        This method updates the dataset type and, if provided, initializes the
        dataset attributes (`dataset_attrs`). It is particularly useful when
        working with **time-series** datasets, where additional metadata is required.

        Args:
            dataset_type (DatasetType | str): new dataset type
            dataset_attrs (dict | None, optional): Dataset attrs for TIMESERIES dataset. Defaults to None.
        """
        dataset_type = DatasetType(dataset_type)
        dataset_attrs = DatasetAttr(**dataset_attrs) if dataset_attrs is not None else None
        self.__set_dataset_type(dataset_type, dataset_attrs)

        if dataset_attrs is not None:
            self.dataset_attrs = dataset_attrs
        else:
            self.dataset_attrs = None

    def set_dataset_attrs(
        self,
        sortby: str | List[str],
        entities: str | List[str] | None = None
    ):
        """Update dataset attributes.

        Args:
            sortby (str | List[str]): Column(s) that express the temporal component
            entities (str | List[str] | None, optional): Column(s) that identify the entities. Defaults to None
        """
        dataset_attrs = {
            "sortbykey": sortby,
        }
        if entities:
            dataset_attrs["entities"] = entities

        dataset_attrs = DatasetAttr(**dataset_attrs)
        self.dataset_attrs = dataset_attrs

    def _process_characteristics(self, dataset: Dataset, schema: dict, characteristics_dictionary: dict | None = None):
        column_tags = {}
        characteristics_dictionary = {
        } if characteristics_dictionary is None else characteristics_dictionary

        if self._manual_characteristics:
            column_tags = self._get_manual_characteristics()

        characteristics_dictionary = self.clean_characteristics(
            characteristics_dictionary,
            threshold=self._characteristics_threshold,
            confidence_level=self._characteristics_confidence_level
        )
        # upddate tags with the automatic characteristics
        for col, tag in characteristics_dictionary.items():
            auto_tags = tuple(t["characteristic"] for t in tag)
            if col in column_tags:
                column_tags[col] = tuple(
                    set([*column_tags[col], *auto_tags])
                )
            else:
                column_tags[col] = auto_tags

        # update dict with the manual characteristics
        for col, tags in column_tags.items():
            default_dict = {
                "value": 1,
                "upper_bound": 1,
                "lower_bound": 1,
            }
            if col in characteristics_dictionary:
                chars = [
                    detail["characteristic"]
                    for detail in characteristics_dictionary[col]
                ]
                for tag in tags:
                    if tag not in chars:
                        characteristics_dictionary[col].append({
                            "characteristic": tag, **default_dict
                        })
            else:
                for tag in tags:
                    characteristics_dictionary[col] = [{
                        "characteristic": tag, **default_dict
                    }]

        return characteristics_dictionary, column_tags

    def clean_characteristics(
        self, matched_dictionary: dict, threshold: float, confidence_level: float
    ) -> dict:
        z_value = determine_z_value(confidence_level)
        summary_dictionary = {}
        sample_sizes = SAMPLE_SIZES[confidence_level]
        for column in matched_dictionary:
            max_key = max(
                matched_dictionary[column], key=matched_dictionary[column].get
            )
            p = matched_dictionary[column][max_key]
            if column in matched_dictionary:
                p, _, upper_bound, lower_bound = calculate_wilson_cc(
                    z=z_value, n=sample_sizes, p=p
                )
            else:
                upper_bound, lower_bound = p, p

            if upper_bound > threshold:
                summary_dictionary[column] = [{
                    "characteristic": max_key,
                    "value": p,
                    "upper_bound": upper_bound,
                    "lower_bound": lower_bound,
                }]
        return summary_dictionary

    def _deduce_characteristics(self):
        """Automatically deduce some characteristics based on the metadata
        information."""
        if self._compute_config.characteristics == ComputeMode.NOW and not self._is_multitable:
            unique_cols = [c.column for c in self.warnings['unique']]
            missings = [c.column for c in self.warnings['missings']]
            ids_cols = [
                c for c in unique_cols
                if (
                    c not in missings
                    and self.columns[c].datatype in [DataType.CATEGORICAL, DataType.STR]
                    and self.columns[c].vartype not in [VariableType.DATE, VariableType.DATETIME]
                )
            ]
            self.add_characteristics(
                characteristics={c: 'id' for c in ids_cols})

    def _imbalance_score(self, value_counts: dict[str, pdSeries]) -> dict:
        class_imbalance = {}
        for col, value_count in value_counts.items():
            if self.columns[col].datatype in CATEGORICAL_DTYPES:
                n_classes = len(value_count)
                if n_classes > 1:
                    class_imbalance[col] = 1 - (
                        entropy(value_count.to_numpy(dtype=int), base=2)/log2(n_classes))
                else:
                    class_imbalance[col] = 0
        return class_imbalance

    def _get_manual_characteristics(self) -> dict:
        tags = self._manual_characteristics
        tags = {
            k: tuple(v) if isinstance(v, Iterable) else tuple([v])
            for k, v in tags.items()
        }
        return tags

    def _infer_datatype(self, column: str, vartype: str | VariableType, cardinality: dict[str, int], word_count: dict = {}) -> DataType:
        if VariableType(vartype) == VariableType.BOOL:
            return DataType.CATEGORICAL
        elif VariableType(vartype) in [VariableType.DATE, VariableType.DATETIME]:
            return DataType.DATE
        elif VariableType(vartype) in [VariableType.INT, VariableType.FLOAT]:
            if cardinality[column] >= self._categorical_threshold or cardinality[column] == 0:
                return DataType.NUMERICAL
            else:
                return DataType.CATEGORICAL
        elif VariableType(vartype) == VariableType.STR:
            if cardinality[column] >= self._categorical_threshold or cardinality[column] == 0:
                if word_count[column] >= self._wordcount_threshold:  # todo revise this value
                    return DataType.LONGTEXT
                else:
                    return DataType.STR
            else:
                return DataType.CATEGORICAL

    def _get_columns_metadata(
        self,
        word_count: dict,
        cardinality: dict[str, int],
        characteristics: dict,
        schema: dict,
        columns: dict | None,
    ) -> dict:
        cols_metadata = {}
        for col, vartype in schema.items():
            datatype = self._infer_datatype(
                col, vartype, cardinality, word_count)
            cols_metadata[col] = Column(
                name=col,
                datatype=datatype,
                vartype=VariableType(vartype),
                characteristics=characteristics.get(col),
            )

        if columns is not None:
            for col, datatype in columns.items():
                cols_metadata[col].datatype = DataType(datatype)

        return cols_metadata

    def _get_warnings(self):
        warnings = {
            WarningType.SKEWNESS.value: SkewnessWarning(),
            WarningType.CARDINALITY.value: HighCardinalityWarning(),
            WarningType.MISSINGS.value: MissingValuesWarning(),
            WarningType.CONSTANT.value: ConstantWarning(),
            WarningType.ZEROS.value: ZerosWarning(),
            WarningType.INFINITY.value: InfinityWarning(),
            WarningType.IMBALANCE.value: ImbalanceWarning(),
            WarningType.UNIQUE.value: UniqueWarning(),
            # TODO improvment for numerical columns uniform distribution validation
            # WarningType.UNIFORM.value: UniformWarning(),
            WarningType.CONSTANT_LENGTH.value: ConstantLengthWarning(),
        }
        if "correlation" in self.summary and isinstance(self.summary["correlation"], pdDataframe):
            warnings[WarningType.CORRELATION.value] = CorrelationWarning()

        if self.__intercolumns_warnings:
            warnings[WarningType.DUPLICATES.value] = DuplicatesWarning()

        engine = WarningEngine(warnings)

        columns = {name: col.datatype for name, col in self.columns.items()}
        return engine.evaluate(self.summary, columns)

    @staticmethod
    def _graph_correlation(dataset: Dataset, vartypes: dict, nrows: int, value_counts: dict[str, pdSeries], max_cardinality: int = 100, min_rows_sampling: int = 1_000_000):
        df = dataset._data
        cols_to_encode = [col for col, type_ in vartypes.items(
        ) if VariableType(type_) == VariableType.STR]

        counts_val = {k: list(value_counts[k].head(
            max_cardinality).index) for k in cols_to_encode}
        encoded = df.copy()
        for k in cols_to_encode:
            encoded[k] = encoded[k].astype('category').cat.set_categories(
                counts_val[k]).astype(df[k].dtype)

        if nrows > min_rows_sampling:
            frac = min_rows_sampling / nrows
            encoded = encoded.sample(frac=frac)

        return association_matrix(encoded, vartypes=vartypes, columns=list(vartypes.keys()))

    @staticmethod
    def _characteristics(schema: dict, nrows: int, no_null, confidence_level: float):
        charac_cols = {col: VariableType(type_) for col, type_ in schema.items(
        ) if VariableType(type_) in [VariableType.STR, VariableType.INT]}

        sample_size = SAMPLE_SIZES[confidence_level]
        charac_sample = {col: no_null[col].sample(
            frac=(sample_size / nrows), replace=True)
            for col in charac_cols.keys()}

        # Keep only string columns
        charac_accessor = {col: charac_sample[col].astype(
            str) for col in charac_cols.keys()
            if charac_cols[col] == VariableType.STR}

        return charac_accessor

    def _graph_base(self, dataset: Dataset) -> tuple[dict, dict]:
        # Get dataset shape
        schema = dataset.schema
        df = dataset._data

        num_cols = [col for col, type_ in schema.items() if VariableType(type_) in [
            VariableType.FLOAT, VariableType.INT]]
        str_cols = [col for col, type_ in schema.items(
        ) if VariableType(type_) == VariableType.STR]

        nrows = self._graph_nrows(dataset)

        # General properties
        no_null = self._graph_no_null(dataset)
        cards = {col: no_null[col].nunique() for col in df.columns}
        missings = {col: nrows - no_null[col].shape[0] for col in df.columns}
        duplicates = df.nunique_approx()

        # Numerical columns only
        num_values = {col: no_null[col] for col in num_cols}
        skewness = {col: skew(num_values[col]) for col in num_cols}
        infinity = {col: isinf(num_values[col]).sum() for col in num_cols}
        zeros = {col: (no_null[col] == 0).sum() for col in num_cols}

        # String columns only
        no_null_str = {col: no_null[col].str for col in str_cols}
        str_len = {col: v.len() for col, v in no_null_str.items()}
        str_len_res = {col: {"max": length.max(), "min": length.min(
        ), "mean": length.mean()} for col, length in str_len.items()}
        word_count = {col: length.count(r"\s+").add(1).mean()
                      for col, length in no_null_str.items()}

        domain_cols = [col for col, type_ in schema.items() if VariableType(type_)
                       in [VariableType.FLOAT, VariableType.INT, VariableType.DATETIME, VariableType.DATE]]

        domains = {col: {'min': no_null[col].min(
        ), 'max': no_null[col].max()} for col in domain_cols}

        val_count_cols = [col for col, type_ in schema.items() if VariableType(
            type_) in [VariableType.BOOL, VariableType.STR, VariableType.INT]]

        value_counts = {col: no_null[col].value_counts()
                        for col in val_count_cols}

        tasks = {
            'nrows': nrows,
            'cardinality': cards,
            'duplicates': duplicates,
            'missings': missings,
            'skewness': skewness,
            'infinity': infinity,
            'zeros': zeros,
            'string_len': str_len_res,
            'word_count': word_count,
            'domains': domains,
            'value_counts': value_counts,
        }

        return tasks, no_null

    def _graph_nrows(self, dataset) -> int:
        return dataset.shape(lazy_eval=False, delayed=True)[0]

    def _graph_no_null(self, dataset: Dataset) -> dict:
        no_null = {col: drop_null(dataset._data[col], is_str=VariableType(
            dataset.schema[col]) == VariableType.STR) for col in dataset._data.columns}
        return no_null

    def _graph_characteristics(self, dataset: Dataset, dependencies: dict | None = None) -> dict:
        schema = dataset.schema
        df = dataset._data
        dependencies = dependencies if dependencies is not None else {}

        # Fast nrows approximation. To be used to determine if sample is needed and the sample size.
        nrows = dependencies.get('nrows', self._graph_nrows(dataset))
        if df.npartitions > 1:
            fast_nrows = (df.partitions[0].shape[0]).compute() * df.npartitions
        else:
            fast_nrows = nrows

        no_null = dependencies.get('no_null', self._graph_no_null(dataset))
        charac_samples = self._characteristics(
            schema=schema, nrows=fast_nrows, no_null=no_null, confidence_level=self._characteristics_confidence_level)
        tasks = {
            'charac_samples': charac_samples
        }
        return tasks

    def _compute_characteristics(self, dataset: Dataset, columns: dict | None = None, dependencies: dict | None = None):
        if not self._base_computed:
            self.__call__(dataset=dataset, columns=columns)

        characteristics_tasks = self._graph_characteristics(
            dataset=dataset, dependencies=dependencies)
        results = compute(characteristics_tasks)[0]
        char_presidio = presidio_detect_characteristics(
            results['charac_samples'], threshold=self._characteristics_threshold)
        del results['charac_samples']

        if 'characteristics' not in results:
            results['characteristics'] = {}
        results['characteristics'].update(char_presidio)

        return self._process_characteristics(dataset, dataset.schema, results.get('characteristics', {}))

    def _compute_characteristics_with_dependencies(self, dataset: Dataset, columns: dict | None = None, deferred: bool = False, dependencies: dict | None = None):
        if not deferred:
            data = self._compute_characteristics(
                dataset=dataset, columns=columns, dependencies=dependencies)
            assign_characteristics(data, self, dataset.schema, columns)
            return self.summary['characteristics']
        else:
            client = DaskCluster().client
            task = client.submit(self._compute_characteristics, dataset)
            task.add_done_callback(lambda x: assign_characteristics(
                x, self, dataset.schema, columns))
            self.status['characteristics'] = task
            return task

    def compute_characteristics(self, dataset: Dataset, columns: dict | None = None, deferred: bool = False) -> dict | Future:
        """Compute the dataset's characteristics.

        The method returns the characteristics and update the metadata instance's summary accordingly.

        Args:
            dataset: dataset corresponding to the Metadata instance
            columns: columns dictionary
            deferred: defer the computation if True, else compute now

        Returns:
            dict if deferred is False, Future otherwise
        """
        return self._compute_characteristics_with_dependencies(dataset=dataset, columns=columns, deferred=deferred)

    def _compute_correlation(self, dataset: Dataset, columns: dict | None = None):
        if not self._base_computed:
            self.__call__(dataset=dataset, columns=columns)
        task_correlation = self._graph_correlation(dataset=dataset,
                                                   vartypes=dataset.schema,
                                                   nrows=self.summary['nrows'],
                                                   value_counts=self.summary['value_counts'],
                                                   max_cardinality=Metadata.MAX_CORR_CARDINALITY,
                                                   min_rows_sampling=Metadata.MIN_ROWS_FOR_SAMPLING)
        return compute(task_correlation)[0]

    def compute_correlation(self, dataset: Dataset, columns: dict | None = None, deferred: bool = False) -> pdDataframe | Future:
        """Compute the dataset's correlation matrix.

        The method returns the correlation matrix and update the metadata instance's summary accordingly.

        Args:
            dataset: dataset corresponding to the Metadata instance
            columns: columns dictionary
            deferred: defer the computation if True, else compute now

        Returns:
            pandas dataframe if deferred is False, Future otherwise
        """
        if not deferred:
            corr = self._compute_correlation(dataset=dataset, columns=columns)
            assign_correlation(corr, self)
            return self.summary['correlation']
        else:
            client = DaskCluster().client
            task = client.submit(self._compute_correlation, dataset)
            task.add_done_callback(lambda x: assign_correlation(x, self))
            self.status['correlation'] = task
            return task

    def _compute_metadata(self, dataset: Dataset, columns: dict | None = None) -> dict:
        with ProgressBar():
            self._metadata["summary"] = {}
            self._cols_dtypes = dataset.schema
            # Get dataset shape
            schema = dataset.schema
            tasks, no_null = self._graph_base(dataset)

            if self._compute_config.characteristics == ComputeMode.NOW:
                characteristics_tasks = self._graph_characteristics(
                    dataset=dataset,
                    dependencies={'nrows': tasks['nrows'], 'no_null': no_null})
                tasks.update(characteristics_tasks)

            logger.debug("[METADATA] - Computing the metadata summary.")
            summary = compute(tasks)[0]

            # Post-process
            self._metadata["nrows"] = tasks['nrows']
            summary['cardinality'] = {
                k: int(v) for k, v in summary['cardinality'].items()}
            summary['duplicates'] = max(
                0, int(summary['nrows'] - summary['duplicates']))

            # Characteristics via Presidio
            if self._compute_config.characteristics == ComputeMode.NOW:
                char_presidio = presidio_detect_characteristics(
                    summary['charac_samples'], threshold=self._characteristics_threshold)
                if "characteristics" not in summary:
                    summary['characteristics'] = {}
                summary['characteristics'].update(char_presidio)
                del summary['charac_samples']
            else:
                summary['characteristics'], tags = {}, {}

            if self._compute_config.characteristics == ComputeMode.NOW or self._manual_characteristics:
                summary['characteristics'], tags = self._process_characteristics(
                    dataset, schema, summary.get('characteristics', {}))

            # Assign the column data types
            self.__columns = self._get_columns_metadata(
                word_count=summary["word_count"],
                cardinality=summary['cardinality'],
                characteristics=tags,
                schema=schema,
                columns=columns
            )

            # Update FLOAT / Categorical to have the value_counts
            col_cat_float = [c for c, v in self.columns.items() if v.vartype == VariableType.FLOAT and v.datatype ==
                             DataType.CATEGORICAL and summary['cardinality'][c] < self._categorical_threshold]
            if col_cat_float:
                no_null = {col: drop_null(dataset._data[col], is_str=VariableType(
                    schema[col]) == VariableType.STR) for col in col_cat_float}
                add_value_counts = {
                    col: no_null[col].value_counts() for col in col_cat_float}
                summary['value_counts'].update(compute(add_value_counts)[0])

            summary["imbalance"] = self._imbalance_score(
                summary['value_counts'])
            self._metadata["summary"].update(summary)

            # Correlation
            correlation_mode = self._compute_config.correlation
            if correlation_mode in [ComputeMode.NOW, ComputeMode.DEFERRED]:
                deferred = correlation_mode == ComputeMode.DEFERRED
                self.compute_correlation(
                    dataset=dataset, columns=columns, deferred=deferred)
            else:
                self._metadata["summary"]['correlation'] = pdDataframe()

            # Characteristics if deferred
            characteristics_mode = self._compute_config.characteristics
            if characteristics_mode == ComputeMode.DEFERRED:
                self._compute_characteristics_with_dependencies(dataset=dataset, columns=columns, deferred=True, dependencies={
                    'nrows': tasks['nrows'], 'no_null': no_null})

            self.__warnings = self._get_warnings()

        return self._metadata

    def __validate_entities(self) -> None:
        if not self._is_multitable:
            invalid_entities_columns = [c for c in self._dataset_attrs.entities if self.columns[c].datatype ==
                                        DataType.NUMERICAL and self.columns[c].vartype == VariableType.FLOAT]
            if invalid_entities_columns:
                raise InvalidEntityColumnError(
                    "The following columns cannot be set as entities because they are NUMERICAL/FLOAT: {}".format(", ".join(invalid_entities_columns)))

    def _add_characteristics_to_summary(self, column: str, characteristic: ColumnCharacteristic):
        tag = {
            "characteristic": characteristic,
            "value": 1,
            "upper_bound": 1,
            "lower_bound": 1,
        }
        characteristics = self.summary["characteristics"]
        if column in characteristics:
            if not any([
                tag["characteristic"] == characteristic
                for tag in characteristics[column]
            ]):
                characteristics[column].append(tag)
        else:
            characteristics[column] = [tag]

    def add_characteristics(self, characteristics: dict[str, list[ColumnCharacteristic | str] | ColumnCharacteristic | str]):
        """Add characteristics to the specified columns.

        The argument `characteristics` is dictionary indexed on the columns that accept two syntaxes:
            1. a characteristic
            2. a list of characteristics

        Example:

            ```python
            characteristics = {
                'col1': 'phone',
                'col2': ['uuid', 'name']
            }
            metadata.add_characteristics(characteristics)
        ```

        Args:
            characteristics (dict[str, list[ColumnCharacteristic | str] | ColumnCharacteristic | str]): characteristics to add
        """
        for column, characteristic in characteristics.items():
            if isinstance(characteristic, list):
                for char in characteristic:
                    self.add_characteristic(column=column, characteristic=char)
            else:
                self.add_characteristic(
                    column=column, characteristic=characteristic)

    def add_characteristic(self, column: str, characteristic: ColumnCharacteristic | str):
        """Add new characteristic to a column.

        Args:
            column (str): column name
            characteristic (ColumnCharacteristic): characteristic to add
        """
        if isinstance(characteristic, str):
            characteristic = ColumnCharacteristic[characteristic.upper()]
        col = self.columns[column]
        col.characteristics = [] if col.characteristics is None else col.characteristics
        col.characteristics = tuple(
            set([*col.characteristics, characteristic])
        )
        self._add_characteristics_to_summary(column, characteristic)

    def remove_characteristics(self, characteristics: dict[str, list[ColumnCharacteristic | str] | ColumnCharacteristic | str]):
        """Remove characteristics to the specified columns.

        The argument `characteristics` is dictionary indexed on the columns that accept two syntaxes:
            1. a characteristic
            2. a list of characteristics

        Example:

            ```python
            characteristics = {
                'col1': 'phone',
                'col2': ['uuid', 'name']
            }
            metadata.remove_characteristics(characteristics)
        ```

        Args:
            characteristics (dict[str, list[ColumnCharacteristic | str] | ColumnCharacteristic | str]): characteristics to add
        """
        for column, characteristic in characteristics.items():
            if isinstance(characteristic, list):
                for char in characteristic:
                    self.remove_characteristic(
                        column=column, characteristic=char)
            else:
                self.remove_characteristic(
                    column=column, characteristic=characteristic)

    def remove_characteristic(self, column: str, characteristic: ColumnCharacteristic | str):
        """Remove a characteristic from a column.

        Args:
            column (str): column name
            characteristic (ColumnCharacteristic): characteristic to remove
        """
        if isinstance(characteristic, str):
            characteristic = ColumnCharacteristic[characteristic.upper()]
        col: Column = self.columns[column]
        col.characteristics = tuple(
            set(col.characteristics) - set([characteristic])
        )

        characteristics = self.summary["characteristics"]

        if column in characteristics:
            idx = None
            for i, tag in enumerate(characteristics[column]):
                if tag["characteristic"] == characteristic:
                    idx = i
                    break
            if idx is not None:
                # remove characteristic
                characteristics[column].pop(idx)
                # if there is no more characteristics remove column
                if len(characteristics[column]) == 0:
                    characteristics.pop(column)

    def get_characteristics(self) -> dict[str, list[ColumnCharacteristic]]:
        """Get the characteristics for all columns.

        Returns:
            dict[str, list[ColumnCharacteristic]]: characteristics dictionary
        """
        chars = {}
        for name, col in self.columns.items():
            chars[name] = [] if col.characteristics is None else list(
                col.characteristics)
        return chars

    def set_characteristics(self, characteristics: dict[str, list[ColumnCharacteristic | str]]):
        """Define the characteristics for all columns.

        Obs.: this will overwrite any previous definition for characteristics

        Args:
            characteristics (dict[str, list[ColumnCharacteristic]]): the new set of characteristics
        """
        for col, tags in characteristics.items():

            if not tags:
                continue

            unique_tags = set()
            for tag in tags:
                tag = tag if isinstance(
                    tag, ColumnCharacteristic) else ColumnCharacteristic[tag.upper()]
                unique_tags.add(tag)
                self._add_characteristics_to_summary(col, tag)

            self.columns[col].characteristics = tuple(unique_tags)

    def __clean_characteristics(self, col_name: str):
        if self.columns[col_name].vartype in [VariableType.DATE, VariableType.DATETIME]:
            self.remove_characteristics({col_name: ['id']})

    def get_possible_targets(self) -> tuple:
        """
        Identify valid target columns for predictive modeling or synthetic data generation.

        This method evaluates the dataset and determines which columns are suitable as target variables.
        Columns are excluded from consideration if they fall into any of the following categories:

        - **Invalid data types** (e.g., long text, string, date).
        - **Constant values** (columns with only one unique value).
        - **ID-like columns** (unique identifiers that do not hold predictive value).
        - **Columns with missing values** (to ensure data integrity)
        - **Columns with defined characteristics**

        Returns:
            targets (tuple): a list with the name of the columns that are potential target variables and its details as a dictionary.

        """
        details = defaultdict(list)

        invalid_columns = set(
            self.longtext_vars
            + self.string_vars
            + self.date_vars
        )

        for col in invalid_columns:
            details[col].append('invalid_data_type')

        # Valid columns target columns are numerical or categorical
        cols = list(set(self.columns) - invalid_columns)

        for col in cols:
            if col in self.isconstant:
                invalid_columns.add(col)
                details[col].append("constant")

            if col in self.id_vars:
                invalid_columns.add(col)
                details[col].append("ID-like")

            val = self._metadata["summary"]["missings"][col]
            if val > 0:
                invalid_columns.add(col)
                details[col].append("missing")

        target_cols = list(set(cols) - invalid_columns)

        return target_cols, details

    @property
    def columns(self) -> dict | None:
        """
         Get the column metadata for the dataset.

        This property returns a dictionary containing metadata about the dataset's columns,
        including feature names and their associated characteristics. It is primarily used
        to provide insights into the structure of the dataset.

        Returns:
            columns (dict): metadata dictionary with the mapping of the columns along with their variable and data types. Returns an object Column for each column.
        """
        return self.__columns

    def update_datatypes(self, value: dict, dataset: Dataset | None = None):
        """Method to update the data types set during the Metadata automatic
        datatype inference.

        Valid datatypes to update the columns are: "longtext",
        "categorical", "numerical", "date" and "id". value (dict): A
        dictionary with the name: datatype value to be assigned to the
        column. Provide only the names of the columns that need a
        datatype update.
        """
        valid_input_col(d=value)
        assert all(col in self.columns.keys() for col in value.keys()), (
            "Please provide a dictionary with valid columns. "
            "The provided columns do not belong to this metadata."
        )
        previous_val = {}
        for col, val in value.items():
            col_def = self.__columns[col]
            previous_val[col] = deepcopy(col_def)
            # Creating a new Column definition so it raises an error when setting an invalid datatype
            vartype = col_def.vartype if dataset is None else dataset.schema.get(
                col, col_def.vartype)
            self.__columns[col] = Column(
                col_def.name, datatype=DataType(val), vartype=VariableType(vartype), characteristics=col_def.characteristics
            )
            self.__clean_characteristics(col)

        if dataset is not None and self._metadata:  # Not computed yet anyway
            new_cat_columns = {k: DataType(val) for k, val in value.items()
                               if DataType(val) == DataType.CATEGORICAL and previous_val[k].datatype != DataType.CATEGORICAL
                               or (DataType(val) == DataType.CATEGORICAL and previous_val[k].datatype == DataType.CATEGORICAL and previous_val[k].vartype != self.__columns[k].vartype)
                               }
            if new_cat_columns:
                schema = {k: previous_val[k].vartype if dataset is None else dataset.schema.get(
                    col, previous_val[k].vartype) for k in new_cat_columns.keys()}
                no_null = {col: drop_null(dataset._data[col], is_str=VariableType(
                    schema[col]) == VariableType.STR) for col in new_cat_columns}
                add_value_counts = {
                    col: no_null[col].value_counts() for col in new_cat_columns}
                self._metadata["summary"]['value_counts'].update(
                    compute(add_value_counts)[0])

        self.__warnings = self._get_warnings()
        self._deduce_characteristics()

    @property
    def cardinality(self) -> dict:
        """
        A property that returns a tuple with a dict with categorical variables approximated
        cardinality and the sum of the total cardinality.

        Returns:
            cardinality (dict): A dictionary with categorical variables approximated cardinality values.
        """
        cardinality_dict = {
            k: int(v)
            for k, v in self._metadata["summary"]["cardinality"].items()
            if k in self.categorical_vars
        }
        return cardinality_dict

    @property
    def isconstant(self) -> list:
        """
        Returns a list with the name of the columns that are constant
        throughout the dataset, i.e., always assume the same value.

        A column is considered constant only and only when the whole columns assume the same value
        the new definition accounting for the missing values also ensures improvements in what concerns
        replicating missing values distribution

        Returns:
            isconstant (list): A list of columns that are constant
        """

        # A column is considered constant only and only when the whole columns assume the same value
        # the new definition accounting for the missing values also ensures improvements in what concerns
        # replicating missing values distribution
        constant_cols = [
            col
            for col, val in self._metadata["summary"]["cardinality"].items()
            if (
                int(val) == 1
                and self._metadata["summary"]['missings'][col] == 0
            ) or (
                self._metadata["summary"]['missings'][col] == self._metadata["summary"]['nrows']
            )
        ]

        return constant_cols

    @property
    def warnings(self) -> dict:
        """
        Get dataset warnings based on statistical and structural analyses.

        This property returns a dictionary of warnings that highlight potential
        **data quality issues**, such as **skewness, cardinality, constant values,
        correlations, imbalances, and constant-length features**. These warnings
        are useful for **profiling, preprocessing, and synthetic data generation**.

        Returns:
            warnings (dict): A dictionary of warnings that highlight potential issues with the dataset variables or columns.

        """
        return self.__warnings

    @property
    def summary(self):
        """
        Get a comprehensive summary of the dataset's metadata.

        This property provides a structured summary containing **key dataset metrics,
        column details, computed statistics, and detected warnings**. It is useful for
        **profiling, data validation, and integration with other libraries**.

        Returns:
            summary (dict): A dictionary containing summary statistics about the dataset. It includes all the calculated metrics such as:
                        - **Dataset Type**: `"TABULAR"`, `"TIME-SERIES"`, etc.
                        - **Number of Columns**: Total feature count.
                        - **Duplicate Rows**: Number of duplicate records detected.
                        - **Target Column**: Identifies a target variable if applicable.
                        - **Column Details**: Data type, variable type, and characteristics for each feature.
                        - **Warnings**: Potential data quality issues such as skewness, cardinality, and imbalances.
        """
        return self._metadata["summary"]

    @property
    def shape(self) -> tuple:
        """
        Get the shape of the dataset that was fed into the Metadata.

        Returns:
            shape (tuple): A tuple containing the shape of the dataset that was fed into the Metadata (nrows, ncols). Is only available if dataset != None.
        """
        return (self._metadata["nrows"], self.ncols)

    @property
    def ncols(self) -> int:
        """
        Get the number of columns in the dataset and/or ConfigurationBuilder

        Returns:
            ncols (int): The number of columns in the dataset.
        """
        return len(self.__columns)

    @property
    def target(self):
        """
        Get the target column in the dataset.
        Returns:
            target (str): The target column in the dataset.
        """
        return self._target

    @target.setter
    def target(self, value: Union[str, Column]):
        """
        Set the target column in the dataset.
        Args:
            value (str): The name of the target column in the dataset.
        """
        if isinstance(value, str):
            try:
                value = self.__columns[value]
            except KeyError:
                raise Exception(
                    f"The column {value} does not exist in the dataset. Please check your input."
                )

        assert value.datatype in [
            DataType.NUMERICAL,
            *CATEGORICAL_DTYPES,
        ], f"The target variable must be of one of the following data type {[DataType.NUMERICAL.value, *[v.value for v in CATEGORICAL_DTYPES]]}."

        self._target = value

    @property
    def numerical_vars(self):
        """
        Return the list of numerical columns in the dataset.
        Returns:
            numerical_cols (list): A list with the name of numerical columns in the dataset.
        """
        return [
            col
            for col, metadata in self.__columns.items()
            if metadata.datatype == DataType.NUMERICAL
        ]

    @property
    def date_vars(self):
        """
        Return the list of date columns in the dataset.
        Returns:
            date_cols (list): A list with the name of date columns in the dataset.
        """
        return [
            col
            for col, metadata in self.__columns.items()
            if metadata.datatype == DataType.DATE
        ]

    @property
    def categorical_vars(self):
        """
        Return the list of categorical columns in the dataset.
        Returns:
            numerical_cols (list): A list with the name of categorical columns in the dataset.
        """
        return [
            col
            for col, metadata in self.__columns.items()
            if metadata.datatype in CATEGORICAL_DTYPES
        ]

    @property
    def id_vars(self):
        """
        Return the list of ID columns in the dataset.
        Returns:
            numerical_cols (list): A list with the name of ID columns in the dataset.
        """
        n_uniquevals = self.summary.get("cardinality", {})
        n_uniquevals = {col: val / self._metadata["nrows"]
                        for col, val in n_uniquevals.items()}
        return [col for col, val in n_uniquevals.items() if (val >= 0.99 and col not in self.numerical_vars)]

    @property
    def longtext_vars(self):
        """
        Return the list of longtext columns in the dataset.
        Returns:
            numerical_cols (list): A list with the name of longtext columns in the dataset.
        """
        return [
            col
            for col, metadata in self.__columns.items()
            if metadata.datatype == DataType.LONGTEXT
        ]

    @property
    def string_vars(self):
        """
        Return the list of string columns in the dataset.
        Returns:
            numerical_cols (list): A list with the name of string columns in the dataset.
        """
        return [
            col
            for col, metadata in self.__columns.items()
            if metadata.datatype == DataType.STR
        ]

    @property
    def dataset_attrs(self):
        """
        A property that returns a dictionary with the defined dataset attributes
        Returns:
            dataset_attrs (dict): a dictionary with the defined dataset attributes
        """
        return self._dataset_attrs

    @dataset_attrs.setter
    def dataset_attrs(self, attrs: dict | DatasetAttr | None = None):
        """
        Method that allows to set the dataset attributes.
        Args:
            attrs (dict): A dictionary with the defined dataset attributes. If dataset_attrs is None, this is expected to clean the Metadata dataset_atts
        """
        if attrs:
            self.__validate_dataset_attrs(attrs, columns=self.columns)
            self._dataset_attrs = attrs
            self._metadata["dataset_attrs"] = self._dataset_attrs
        else:
            self._dataset_attrs = None
            self._metadata["dataset_attrs"] = None


    def save(self, path: str):
        """
        Save the `Metadata` object to a pickle file.

        This method serializes the `Metadata` object and saves it as a **pickle (`.pkl`) file**
        at the specified path. The saved file can later be loaded to restore the metadata
        without reprocessing the dataset.

        Args:
            path (str): The path to save the metadata to. The file extension should be `.pkl` to ensure proper deserialization.

        Returns:
            None: The metadata object is stored in the specified file location.

        ## Example Usage:
        ```python
        from ydata.metadata import Metadata

        # Load dataset metadata
        metadata = Metadata(dataset=my_dataset)

        # Save metadata to a file
        metadata.save("metadata.pkl")
        ```
        """
        try:
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
        """
        Load a `Metadata` object from a saved file.

        This method restores a previously saved `Metadata` object from a **pickle (`.pkl`) file**.
        It allows users to reload metadata without needing to reprocess the dataset.

        Args:
            path (str): The path to load the metadata from.

        Returns:
            metadata (Metadata): A loaded `Metadata` object.

        ## Example Usage:
        ```python
        from ydata.metadata import Metadata

        # Load metadata from a saved file
        metadata = Metadata.load("metadata.pkl")

        # Access dataset insights from loaded metadata
        print(metadata.shape)
        print(metadata.schema)
        print(metadata.summary)
        ```
        """
        try:
            with open(path, "rb") as handle:
                metadata = load(handle)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"The path: '{path}', could not be found. Please provide an existing load path."
            )
        assert isinstance(
            metadata, Metadata
        ), "The provided path does not refer to a Metadata object. Please \
            verify your input path."

        metadata.__migrate()

        return metadata

    def __migrate(self):
        """Define how to migrate metadata pickle from previous versions of the
        library to the latest one."""
        # Introduced in 1.29.0
        cf = getattr(self, '_compute_config', None)
        if cf is None:
            val = self.__init_compute_config(
                config=None, pairwise_metrics=self._pairwise_metrics, infer_characteristics=self._infer_characteristics)
            setattr(self, '_compute_config', val)
        bc = getattr(self, '_base_computed', None)
        if bc is None:
            setattr(self, '_base_computed', 'nrows' in self.summary)

    def _filter_warnings(self, key, updt_meta: Metadata) -> None:
        updt_meta.__warnings = {
            warning: [] for warning, _ in updt_meta.warnings.items()
        }
        for warn_type, value in self.warnings.items():
            for warning in value:
                if warning.column in key:
                    updt_meta.__warnings[warn_type].append(warning)

    def _select_columns(self, key):
        updt_meta = deepcopy(self)
        updt_meta.__columns = {
            col: value for col, value in updt_meta.__columns.items() if col in key
        }

        updt_meta._target = updt_meta._target if updt_meta._target in key else None
        for prop, item in updt_meta._metadata.items():
            if isinstance(item, dict):
                if prop == "summary":
                    for sum_property in [
                        "missings",
                        "skewness",
                        "string_len",
                        "word_count",
                        "cardinality",
                        "domains",
                        "value_counts",
                        "characteristics",
                        "imbalance",
                        "zeros",
                        "infinity"
                    ]:
                        updt_meta._metadata["summary"][sum_property] = {
                            k: val
                            for k, val in updt_meta._metadata["summary"][
                                sum_property
                            ].items()
                            if k in key
                        }
                if 'correlation' in updt_meta._metadata["summary"]:
                    cols = [
                        k for k in key if k in updt_meta._metadata["summary"]['correlation'].columns]
                    updt_meta._metadata["summary"]['correlation'] = updt_meta._metadata["summary"]['correlation'][cols].T[cols]
        self._filter_warnings(key, updt_meta)
        return updt_meta

    def __getitem__(self, key):
        """
        Usage:
        >>> data[ ['columnA', 'columnB'] ]
        """
        return self._select_columns(key)

    def __build_repr(self) -> dict:
        metadata = {
            "Dataset type": self.dataset_type.name,
            "Dataset attributes": [
                {"Attribute": k, "Columns": v}
                for k, v in asdict(self.dataset_attrs).items()
            ]
            if self._dataset_attrs is not None
            else self._dataset_attrs,
            "Number of columns": len(self.__columns) if self.__columns is not None else 0,
            "Number of rows": int(self._metadata.get("summary", {}).get("nrows", 0)),
            "Duplicate rows": int(self._metadata.get("summary", {}).get("duplicates", 0)),
            "Target column": self._target,
        }
        if self.columns is not None:
            warnings = pdDataframe(
                [
                    {"Warning": k, "Columns": [
                        warning.column for warning in v]}
                    for k, v in self.warnings.items()
                    if len(v) > 0
                ]
            )
            metadata.update({
                "Column detail": pdDataframe(
                    [
                        {
                            "Column": k,
                            "Data type": v.datatype.value,
                            "Variable type": v.vartype.value,
                            "Characteristics": ", ".join([c.value for c in v.characteristics])
                        }
                        for k, v in self.columns.items()
                    ]
                ),
                "Warnings": warnings if warnings.shape[0] else "\nNo warning!"
            })
        return metadata

    def __format_repr(self):
        pretty_summary = self.__build_repr()
        str_repr = ""
        for k, val in pretty_summary.items():
            if isinstance(val, pdDataframe):
                str_repr += "\n"
            str_repr += TextStyle.BOLD + f"{k}: " + TextStyle.END
            if type(val) in [str, int, float]:
                str_repr += str(val)
            elif isinstance(val, pdDataframe):
                str_repr += "\n"
                str_repr += val.to_string()
            str_repr += "\n"
        return str_repr

    def __str__(self):
        str_repr = TextStyle.BOLD + "Metadata Summary \n \n" + TextStyle.END
        return str_repr + self.__format_repr()

    @log_time_factory(logger)
    def combine(self, other: Metadata):
        # validate column intersection
        col_intersection = set(self.columns.keys()) & set(other.columns.keys())
        assert len(col_intersection) == 0, \
            f"combine does not support metadata with columns with same name: {col_intersection}"

        # merge columns
        self.__columns.update(other.__columns)

        for k in self.summary:
            if isinstance(self.summary[k], dict):
                self.summary[k].update(other.summary[k])

        # merge warnings
        for k in self.warnings:
            if k in other.warnings:
                self.warnings[k].extend(other.warnings[k])
        for k in other.warnings:
            if k not in self.warnings:
                self.warnings[k] = other.warnings[k]

        # merge target
        if self.target and other.target:
            if self.target.name != other.target.name:
                raise ValueError(
                    f"Metadata objects have different target columns: {self.target.name} and {other.target.name}")
        elif other.target:
            self.target = other.target

        # datasets attrs
        if (
            (self.dataset_attrs and not self.dataset_attrs.empty())
            and (other.dataset_attrs and not other.dataset_attrs.empty())
        ):
            intesection = set(self.dataset_attrs.sortbykey) & set(
                other.dataset_attrs.sortbykey)
            assert len(intesection) == len(self.dataset_attrs.sortbykey), \
                f"Metadata combine with different sortbykey's {self.dataset_attrs.sortbykey}, {other.dataset_attrs.sortbykey}"

            intesection = set(self.dataset_attrs.entities) & set(
                other.dataset_attrs.entities)
            assert len(intesection) == len(self.dataset_attrs.entities), \
                f"Metadata combine with different entities' {self.dataset_attrs.entities}, {other.dataset_attrs.entities}"

        elif other.dataset_attrs and not other.dataset_attrs.empty():
            self._dataset_attrs = other.dataset_attrs

    def __build_metadata_from_config(self, builder: MetadataConfigurationBuilder):
        # init metadata fields
        self.__columns = {}
        self.__warnings = {}
        self._metadata["summary"] = {}
        self.summary["is_from_config"] = True
        self.summary["nrows"] = 0
        for _field in [
            "domains", "cardinality", "missings", "value_counts",
            "skewness", "string_len", "word_count", "characteristics",
            "imbalance", "zeros", "infinity", "extra_data"
        ]:
            self.summary[_field] = {}

        # init columns data
        keywords = {"datatype", "vartype", "characteristic"}
        for col_name, params in builder.config.items():
            self.__columns[col_name] = Column(
                name=col_name,
                datatype=params["datatype"],
                vartype=params["vartype"],
            )
            if "characteristic" in params:
                self.add_characteristic(col_name, params["characteristic"])
            for param, value in params.items():
                if param not in keywords:
                    if col_name not in self.summary["extra_data"]:
                        self.summary["extra_data"][col_name] = {}
                    self.summary["extra_data"][col_name][param] = value

        # build the required statistics
        for col in self.columns.values():
            # domains
            if "min" in builder.config[col.name] or "max" in builder.config[col.name]:
                self.summary["domains"][col.name] = {}
                if "min" in builder.config[col.name]:
                    self.summary["domains"][col.name]["min"] = builder.config[col.name]["min"]
                if "max" in builder.config[col.name]:
                    self.summary["domains"][col.name]["max"] = builder.config[col.name]["max"]

            # missings
            self.summary["missings"][col.name] = builder.config[col.name].get(
                "missings", 0)

            # cardinality
            if "cardinality" in builder.config[col.name]:
                self.summary["cardinality"][col.name] = builder.config[col.name]["cardinality"]

            if "categories" in builder.config[col.name]:
                # new entry added to metadata
                categories = builder.config[col.name]["categories"]
                value_counts = pdDataframe.from_dict([categories]).T[0]
                self.summary["value_counts"][col.name] = value_counts
                self.summary["cardinality"][col.name] = len(categories)
