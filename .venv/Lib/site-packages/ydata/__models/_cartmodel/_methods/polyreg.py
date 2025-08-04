"""File to define the polyregression method."""
from typing import Dict

from numpy import cumsum, sum
from numpy.random import default_rng
from pandas import Series
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from ydata.__models._cartmodel._methods import BaseMethod
from ydata.__models._cartmodel._methods.utils import proper
from ydata.dataset.dataset import Dataset
from ydata.utils.data_types import CATEGORICAL_DTYPES, DataType
from ydata.utils.random import RandomSeed


class PolyregMethod(BaseMethod):
    def __init__(
        self,
        y_dtype: DataType,
        proper: bool = False,
        random_state: RandomSeed = None,
        *args,
        **kwargs
    ):
        """Initialize PolyregMethod.

        Args:
            y_dtype (DataType): Target datatype
            proper (bool): True if proper synthesization
            random_state (int): Internal random state
        """
        self.y_dtype = y_dtype
        self.proper = proper
        self.random_state = random_state

        assert self.y_dtype in CATEGORICAL_DTYPES
        # Specify solver and multi_class to silence this warning
        self.y_encoder = None
        self.polyreg = LogisticRegression(
            solver="lbfgs",
            multi_class="auto",
            max_iter=1000
        )

    def fit(self, X: Dataset, y: Dataset, dtypes: Dict = None, *args, **kwargs):
        """Fit PolyregMethod.

        Args:
            X (Dataset): Predictors
            y (Dataset): Target
            dtypes (Dict): Datatypes of predictors
        """
        if self.proper:
            X, y = proper(X=X, y=y, random_state=self.random_state)

        X, y = self.prepare_dfs(X=X, y=y, dtypes=dtypes,
                                normalise_num_cols=True)

        # Categorical with bool dtype will result in ValueError: Unknown label type: 'unknown' during logreg.fit
        if sorted(list(y.unique())) == sorted([False, True]):
            y = y.astype(int)
        # Categorical with float dtype will result inValueError: Unknown label type: 'continuous' during logreg.fit
        elif y.dtype == "category":
            self.y_encoder = LabelEncoder()
            self.y_encoder.fit(y.cat.categories)
            y = Series(self.y_encoder.transform(y))

        X = X.to_numpy()
        y = y.to_numpy()
        self.polyreg.fit(X, y)

    def predict(self, X_test, dtypes: Dict = None, random_state: RandomSeed = None):
        """Predict using a fitted PolyregMethod.

        Args:
            X_test (Dataset): Predictors to test
            dtypes (Dict): Datatypes of predictors

        Returns:
            y_pred (np.array): Synthesized data
        """
        rng = default_rng(seed=random_state)
        X_test, _ = self.prepare_dfs(
            X=X_test, dtypes=dtypes, normalise_num_cols=True, fit=False
        )
        n_test_rows = len(X_test)

        X_test = X_test.to_numpy()
        y_pred_proba = self.polyreg.predict_proba(X_test)

        uniform_noise = rng.uniform(size=[n_test_rows, 1])
        indices = sum(uniform_noise > cumsum(
            y_pred_proba, axis=1), axis=1).astype(int)
        y_pred = self.polyreg.classes_[indices]
        if self.y_encoder:
            y_pred = self.y_encoder.inverse_transform(y_pred)
        return y_pred
