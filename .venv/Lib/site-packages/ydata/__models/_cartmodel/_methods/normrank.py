"""File to define the logic for the Rank Normalization Method."""
from typing import Dict

from numpy import clip, matmul, max, min, random, sort, sqrt, sum
from numpy.linalg import cholesky, inv
from scipy.stats import norm, rankdata

from ydata.__models._cartmodel._methods import NormMethod
from ydata.__models._cartmodel._methods.utils import smooth


class NormRankMethod(NormMethod):
    """Adapted from norm by carrying out regression on Z scores from ranks
    predicting new Z scores and then transforming back."""

    def fit(self, X, y, dtypes: Dict = None):
        X, y = self.prepare_dfs(X=X, y=y, dtypes=dtypes,
                                normalise_num_cols=True)
        y_real_min, y_real_max = min(y), max(y)
        self.n_rows, n_cols = X.shape

        X = X.to_numpy()
        y = y.to_numpy()
        z = norm.ppf(rankdata(y).astype(int) / (self.n_rows + 1))
        self.norm.fit(X, z)

        residuals = z - self.norm.predict(X)

        if self.proper:
            # Todo revise proper for normalrankmethod

            # Looks like proper is not working quite yet as it produces negative values for a strictly possitive column

            # Draws values of beta and sigma for Bayesian linear regression synthesis of y given x according to Rubin p.167
            # https://link.springer.com/article/10.1007/BF02924688
            self.sigma = sqrt(
                sum(residuals**2) / random.chisquare(self.n_rows - n_cols)
            )
            # NOTE: I don't like the use of inv()
            V = inv(matmul(X.T, X))
            self.norm.coef_ += matmul(
                cholesky(
                    (V + V.T) / 2), random.normal(scale=self.sigma, size=n_cols)
            )
        else:
            self.sigma = sqrt(sum(residuals**2) / (self.n_rows - n_cols - 1))

        if self.smoothing:
            y = smooth(self.dtype, y, y_real_min, y_real_max)

        self.y_sorted = sort(y)

    def predict(self, X_test, dtypes: Dict = None):
        X_test, _ = self.prepare_dfs(
            X=X_test, dtypes=dtypes, normalise_num_cols=True, fit=False
        )
        n_test_rows = len(X_test)

        X_test = X_test.to_numpy()
        z_pred = self.norm.predict(X_test) + random.normal(
            scale=self.sigma, size=n_test_rows
        )
        y_pred_indices = (norm.pdf(z_pred) * (self.n_rows + 1)).astype(int)
        y_pred_indices = clip(y_pred_indices, 1, self.n_rows)
        y_pred = self.y_sorted[y_pred_indices]

        return y_pred
