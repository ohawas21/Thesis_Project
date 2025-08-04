from _typeshed import Incomplete
from pandas import DataFrame as pdDataFrame
from ydata.dataset.multidataset import MultiDataset as MultiDataset
from ydata.metadata import Metadata as Metadata
from ydata.synthesizers.calculated_features import CalculatedFeature

logger: Incomplete

def init_calculated_features(calculated_features: dict | None):
    """Initialize calculated features.

    Args:
        calculated_features (dict | None): calculated feaures configuration.

    Returns:
        list(CalculatedFeature): list of calculated features.
    """
def validate_calculated_features(data: MultiDataset, calculated_features: list[CalculatedFeature]) -> None:
    """Validate calculated features definition.

    Raises ValueError if any incossistent calculated features is found.

    Args:
        data (MultiDataset): dataset were the calculated features will be applied.
        calculated_features (list[CalculatedFeature]): list of calculated features.

    Raises:
        ValueError: error indication the invalid calculated feature
    """
def drop_calculated_features_columns(calculated_features: list[CalculatedFeature], tables_df: dict[str, pdDataFrame]):
    """Remove calculated features columns from the data.

    Obs. modify `tables_df` inplace.

    Args:
        calculated_features (list[CalculatedFeature]): list of calculated features.
        tables_df (dict[str, pdDataFrame]): data were the calculated features will be applied.

    Returns:
        dict[str, pdDataFrame]: tables' data with columns removed
    """
def drop_table_calculated_features_columns(table: str, table_data: pdDataFrame, table_metadata: Metadata, calculated_features: list[CalculatedFeature]):
    """Remove calculated features columns from the data.

    Args:
        calculated_features (list[CalculatedFeature]): list of calculated features.

    Returns:
        dict[str, pdDataFrame]: tables' data with columns removed
    """
def get_tables_from_columns(columns: list[str]) -> set[str]: ...
def is_intra_table_calculated_feature(calculated_feature: CalculatedFeature) -> bool: ...
def apply_table_calculated_features(calculated_features: list[CalculatedFeature], table: str, table_data: pdDataFrame, sample_tables: dict[str, pdDataFrame]) -> pdDataFrame: ...
