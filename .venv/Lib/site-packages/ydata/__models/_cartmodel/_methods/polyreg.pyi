from _typeshed import Incomplete
from ydata.__models._cartmodel._methods import BaseMethod
from ydata.dataset.dataset import Dataset as Dataset
from ydata.utils.data_types import DataType as DataType
from ydata.utils.random import RandomSeed as RandomSeed

class PolyregMethod(BaseMethod):
    y_dtype: Incomplete
    proper: Incomplete
    random_state: Incomplete
    y_encoder: Incomplete
    polyreg: Incomplete
    def __init__(self, y_dtype: DataType, proper: bool = False, random_state: RandomSeed = None, *args, **kwargs) -> None:
        """Initialize PolyregMethod.

        Args:
            y_dtype (DataType): Target datatype
            proper (bool): True if proper synthesization
            random_state (int): Internal random state
        """
    def fit(self, X: Dataset, y: Dataset, dtypes: dict = None, *args, **kwargs):
        """Fit PolyregMethod.

        Args:
            X (Dataset): Predictors
            y (Dataset): Target
            dtypes (Dict): Datatypes of predictors
        """
    def predict(self, X_test, dtypes: dict = None, random_state: RandomSeed = None):
        """Predict using a fitted PolyregMethod.

        Args:
            X_test (Dataset): Predictors to test
            dtypes (Dict): Datatypes of predictors

        Returns:
            y_pred (np.array): Synthesized data
        """
