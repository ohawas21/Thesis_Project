from _typeshed import Incomplete
from enum import Enum
from pandas import DataFrame as pdDataFrame
from ydata.utils.data_types import DataType

class DiscretizationType(Enum):
    UNIFORM = 'uniform'
    QUANTILE = 'quantile'

class Discretizer:
    """A class which enables the discretization of a pandas dataframe. Perform
    this action when you want to convert a continuous variable into a
    categorical variable.

    Attributes:

    method (DiscretizationType): this attribute controls how the buckets of your
    discretization are formed. A uniform discretization type forms the bins to be of
    equal width whereas a quantile discretization type forms the bins to be of equal size.

    n_bins (int): number of bins
    reset_index (bool): instruction to reset the index of the dataframe after the discretization
    """
    discretization_type: Incomplete
    n_bins: Incomplete
    reset_index: Incomplete
    def __init__(self, method=..., n_bins: int = 10, reset_index: bool = False) -> None: ...
    def discretize_dataframe(self, dataframe: pdDataFrame, data_types: dict[str, DataType] | None = None) -> pdDataFrame:
        """discretize the dataframe.

        Args:
            dataframe (pdDataFrame): dataframe to be discretized
            data_types (Optional[dict], optional): dictionary specifying datatypes. Defaults to None.

        Returns:
            pdDataFrame: discretized dataframe
        """
