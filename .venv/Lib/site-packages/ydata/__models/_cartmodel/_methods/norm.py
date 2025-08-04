"""File to define the Norm Method."""
from typing import Dict

from numpy import matmul, max, min, sqrt, sum
from numpy.linalg import cholesky, inv
from numpy.random import default_rng
from sklearn.linear_model import Ridge

from ydata.__models._cartmodel._methods import BaseMethod
from ydata.__models._cartmodel._methods.utils import smooth
from ydata.utils.data_types import DataType
from ydata.utils.random import RandomSeed


class NormMethod(BaseMethod):
    def __init__(
        self,
        y_dtype: DataType,
        smoothing: bool = False,
        proper: bool = False,
        random_state: RandomSeed = None,
        ridge: float = 0.00001,
        *args,
        **kwargs
    ):
        """
        proper: bool. For proper synthesis (proper=TRUE) a CART model is fitted to a bootstrapped sample of the original data
        smoothing: bool. To define whether smoothing should be applied to numerical variables
        """
        self.y_dtype = y_dtype
        self.smoothing = smoothing
        self.proper = proper
        self.random_state = random_state
        self.alpha = ridge
        assert self.y_dtype in [DataType.NUMERICAL, DataType.DATE]
        self.norm = Ridge(alpha=self.alpha, random_state=self.random_state)

    def fit(self, X, y, dtypes: Dict = None, *args, **kwargs):
        rng = default_rng(seed=self.random_state)
        X, y = self.prepare_dfs(X=X, y=y, dtypes=dtypes,
                                normalise_num_cols=True)
        self.y_real_min, self.y_real_max = min(y), max(y)
        n_rows, n_cols = X.shape

        X = X.to_numpy()
        y = y.to_numpy()
        self.norm.fit(X, y)

        residuals = y - self.norm.predict(X)

        if self.proper:
            # looks like proper is not working quite yet as it produces negative values for a strictly possitive column

            # Draws values of beta and sigma for Bayesian linear regression synthesis of y given x according to Rubin p.167
            # https://link.springer.com/article/10.1007/BF02924688
            self.sigma = sqrt(sum(residuals**2) /
                              rng.chisquare(n_rows - n_cols))
            # NOTE: I don't like the use of inv()
            V = inv(matmul(X.T, X))
            self.norm.coef_ += matmul(
                cholesky(
                    (V + V.T) / 2), rng.normal(scale=self.sigma, size=n_cols)
            )
        else:
            self.sigma = sqrt(sum(residuals**2) / (n_rows - n_cols - 1))

    def predict(self, X_test, dtypes: Dict = None, random_state: RandomSeed = None):
        rng = default_rng(seed=random_state)
        X_test_df, _ = self.prepare_dfs(
            X=X_test, dtypes=dtypes, normalise_num_cols=True, fit=False
        )
        n_test_rows = len(X_test_df)

        X_test = X_test_df.to_numpy()
        y_pred = self.norm.predict(X_test) + rng.normal(
            scale=self.sigma, size=n_test_rows
        )

        if self.smoothing:
            y_pred = smooth(self.dtype, y_pred,
                            self.y_real_min, self.y_real_max)

        return y_pred
