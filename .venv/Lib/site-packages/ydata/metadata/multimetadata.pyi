from _typeshed import Incomplete
from ydata.dataset import DatasetType, MultiDataset
from ydata.dataset.schemas.datasets_schema import MultiTableSchema
from ydata.metadata import Metadata

logger: Incomplete

class MultiMetadata:
    dataset_type: Incomplete
    schema: Incomplete
    def __init__(self, multiset: MultiDataset, tables_metadata: dict[str, Metadata] | None = None, dataset_attrs: dict[str, dict] | None = None, dataset_type: dict[str, DatasetType | str] | None = None, schema: MultiTableSchema | None = None) -> None: ...
    @property
    def warnings(self) -> dict: ...
    def compute(self):
        """Request all the tables that are not available yet."""
    def __getitem__(self, key: str) -> Metadata: ...
    def items(self): ...
    def keys(self): ...
    def values(self): ...
    def __iter__(self): ...
    def __delitem__(self, table: str): ...
    def save(self, path: str):
        """Creates a pickle of the metadata object stored in the provided path."""
    @staticmethod
    def load(path: str) -> Metadata:
        """Loads a metadata object from a path to a pickle."""
    def validate_schema(self, other_metadata: MultiMetadata, check_referential_integrity: bool, multiset: MultiDataset | None = None, other_multiset: MultiDataset | None = None) -> dict: ...
    def is_same_schema(self, other_metadata: MultiMetadata, multiset: MultiDataset, other_multiset: MultiDataset) -> bool:
        """Return True is both schema are strictly similar, else False."""
    def get_schema_validation_summary(self, other_metadata: MultiMetadata, multiset: MultiDataset, other_multiset: MultiDataset) -> str: ...
    def set_table_dataset_type(self, table_name: str, dataset_type: str | DatasetType, dataset_attrs: dict | None = None):
        """Update table's metadata dataset type.

        Args:
            table_name (str): Table that will have the dataset type updated
            dataset_type (str | DatasetType): new dataset type
            dataset_attrs (dict | None, optional): Dataset attrs for TIMESERIES dataset. Defaults to None.

        Raises:
            KeyError: when MultiMetadata does not contain the any table named as {table_name}
        """
    def set_table_dataset_attrs(self, table_name: str, sortby: str | list, entities: str | list | None = None):
        """Update table's metadata dataset attributes.

        Args:
            table_name (str): Table that will have the dataset attributes updated
            sortby (str | List[str]): Column(s) that express the temporal component
            entities (str | List[str] | None, optional): Column(s) that identify the entities. Defaults to None

        Raises:
            KeyError: when MultiMetadata does not contain the any table named as {table_name}
        """
