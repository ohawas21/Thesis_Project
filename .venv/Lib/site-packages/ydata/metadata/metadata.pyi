from _typeshed import Incomplete
from dask.distributed import Future
from dataclasses import dataclass
from pandas import DataFrame as pdDataframe
from warnings import warn as warn
from ydata.characteristics import ColumnCharacteristic
from ydata.dataset import Dataset, DatasetType
from ydata.metadata.builder import MetadataConfigurationBuilder
from ydata.metadata.column import Column
from ydata.metadata.compute import ComputeConfig

logger: Incomplete
DEFAULT_PARTITION_SIZE: int

def assign_correlation(data: Future | pdDataframe, m: Metadata): ...
def assign_characteristics(data: Future | tuple, m: Metadata, schema, columns: dict | None = None): ...

@dataclass
class DatasetAttr:
    sortbykey: list[str] = ...
    entities: list[str] = ...
    def __init__(self, sortbykey: list[str] | str, entities: list[str] | str | None = None) -> None: ...
    @staticmethod
    def fields(): ...
    def empty(self): ...

def istype(d: dict, inputtype=...):
    """Validate wether all the values from a dict are of a provided type."""
def valid_input_col(d: dict):
    """Validate input column."""

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
    DATASET_WARNINGS: list[str]
    MIN_ROWS_FOR_SAMPLING: int
    MAX_CORR_CARDINALITY: int
    dataset_type: Incomplete
    status: Incomplete
    def __init__(self, dataset: Dataset | None = None, dataset_attrs: dict | None = None, columns: dict | None = None, dataset_type: DatasetType | str = ..., infer_characteristics: bool = False, characteristics: dict | None = None, pairwise_metrics: bool = True, partition_size: int = ..., intercolumns_warnings: bool = True, compute_config: ComputeConfig | dict | None = None, configuration_builder: MetadataConfigurationBuilder | None = None, partition: bool = False) -> None:
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
    def __call__(self, dataset: Dataset, dataset_attrs: dict | DatasetAttr | None = None, columns: dict | None = None, dataset_type: DatasetType | str = ...) -> Metadata: ...
    def set_dataset_type(self, dataset_type: DatasetType | str, dataset_attrs: dict | None = None):
        """
        Update the dataset type and optionally set dataset attributes.

        This method updates the dataset type and, if provided, initializes the
        dataset attributes (`dataset_attrs`). It is particularly useful when
        working with **time-series** datasets, where additional metadata is required.

        Args:
            dataset_type (DatasetType | str): new dataset type
            dataset_attrs (dict | None, optional): Dataset attrs for TIMESERIES dataset. Defaults to None.
        """
    def set_dataset_attrs(self, sortby: str | list[str], entities: str | list[str] | None = None):
        """Update dataset attributes.

        Args:
            sortby (str | List[str]): Column(s) that express the temporal component
            entities (str | List[str] | None, optional): Column(s) that identify the entities. Defaults to None
        """
    def clean_characteristics(self, matched_dictionary: dict, threshold: float, confidence_level: float) -> dict: ...
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
    def add_characteristic(self, column: str, characteristic: ColumnCharacteristic | str):
        """Add new characteristic to a column.

        Args:
            column (str): column name
            characteristic (ColumnCharacteristic): characteristic to add
        """
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
    def remove_characteristic(self, column: str, characteristic: ColumnCharacteristic | str):
        """Remove a characteristic from a column.

        Args:
            column (str): column name
            characteristic (ColumnCharacteristic): characteristic to remove
        """
    def get_characteristics(self) -> dict[str, list[ColumnCharacteristic]]:
        """Get the characteristics for all columns.

        Returns:
            dict[str, list[ColumnCharacteristic]]: characteristics dictionary
        """
    def set_characteristics(self, characteristics: dict[str, list[ColumnCharacteristic | str]]):
        """Define the characteristics for all columns.

        Obs.: this will overwrite any previous definition for characteristics

        Args:
            characteristics (dict[str, list[ColumnCharacteristic]]): the new set of characteristics
        """
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
    def update_datatypes(self, value: dict, dataset: Dataset | None = None):
        '''Method to update the data types set during the Metadata automatic
        datatype inference.

        Valid datatypes to update the columns are: "longtext",
        "categorical", "numerical", "date" and "id". value (dict): A
        dictionary with the name: datatype value to be assigned to the
        column. Provide only the names of the columns that need a
        datatype update.
        '''
    @property
    def cardinality(self) -> dict:
        """
        A property that returns a tuple with a dict with categorical variables approximated
        cardinality and the sum of the total cardinality.

        Returns:
            cardinality (dict): A dictionary with categorical variables approximated cardinality values.
        """
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
    @property
    def summary(self):
        '''
        Get a comprehensive summary of the dataset\'s metadata.

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
        '''
    @property
    def shape(self) -> tuple:
        """
        Get the shape of the dataset that was fed into the Metadata.

        Returns:
            shape (tuple): A tuple containing the shape of the dataset that was fed into the Metadata (nrows, ncols). Is only available if dataset != None.
        """
    @property
    def ncols(self) -> int:
        """
        Get the number of columns in the dataset and/or ConfigurationBuilder

        Returns:
            ncols (int): The number of columns in the dataset.
        """
    @property
    def target(self):
        """
        Get the target column in the dataset.
        Returns:
            target (str): The target column in the dataset.
        """
    @target.setter
    def target(self, value: str | Column):
        """
        Set the target column in the dataset.
        Args:
            value (str): The name of the target column in the dataset.
        """
    @property
    def numerical_vars(self):
        """
        Return the list of numerical columns in the dataset.
        Returns:
            numerical_cols (list): A list with the name of numerical columns in the dataset.
        """
    @property
    def date_vars(self):
        """
        Return the list of date columns in the dataset.
        Returns:
            date_cols (list): A list with the name of date columns in the dataset.
        """
    @property
    def categorical_vars(self):
        """
        Return the list of categorical columns in the dataset.
        Returns:
            numerical_cols (list): A list with the name of categorical columns in the dataset.
        """
    @property
    def id_vars(self):
        """
        Return the list of ID columns in the dataset.
        Returns:
            numerical_cols (list): A list with the name of ID columns in the dataset.
        """
    @property
    def longtext_vars(self):
        """
        Return the list of longtext columns in the dataset.
        Returns:
            numerical_cols (list): A list with the name of longtext columns in the dataset.
        """
    @property
    def string_vars(self):
        """
        Return the list of string columns in the dataset.
        Returns:
            numerical_cols (list): A list with the name of string columns in the dataset.
        """
    @property
    def dataset_attrs(self):
        """
        A property that returns a dictionary with the defined dataset attributes
        Returns:
            dataset_attrs (dict): a dictionary with the defined dataset attributes
        """
    @dataset_attrs.setter
    def dataset_attrs(self, attrs: dict | DatasetAttr | None = None):
        """
        Method that allows to set the dataset attributes.
        Args:
            attrs (dict): A dictionary with the defined dataset attributes. If dataset_attrs is None, this is expected to clean the Metadata dataset_atts
        """
    def save(self, path: str):
        '''
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
        '''
    @staticmethod
    def load(path: str) -> Metadata:
        '''
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
        '''
    def __getitem__(self, key):
        """
        Usage:
        >>> data[ ['columnA', 'columnB'] ]
        """
    def combine(self, other: Metadata): ...
