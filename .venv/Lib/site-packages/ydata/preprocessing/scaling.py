"""Numerical scaling utilities."""
from collections import defaultdict
from typing import Optional

import joblib
from sklearn.base import BaseEstimator, TransformerMixin

from ydata.utils.timeseries import add_temporal_features


class NumericalFactors(BaseEstimator, TransformerMixin):
    """Transform numerical data based on factors.

    Factors are multiplicative scalars fitted/applied per numerical
    feature per each groupby value.
    """

    def __init__(
        self, groupby: str, factors: Optional[dict] = None, default: bool = True
    ):
        """
        Args:
            groupby (str): column in data used to map the factors
            factors (dict, optional): pre-trained factors to apply.
            default (bool, optional): convert factors to default dict of 1x factor on never-observed groupby values.
        """
        self.groupby = groupby
        self.default = default
        self._set_factors(
            factors
        )  # dict of {feature1: {groupby_value1: factor1, ... }}.

    def _set_factors(self, factors):
        factors = {} if factors is None else factors
        if self.default:
            for feature in factors.keys():
                factors[feature] = defaultdict(lambda: 1, factors[feature])
        self.factors = factors

    def __validate_input(self, X):
        X = X.copy()
        if self.groupby not in X.columns:
            X = add_temporal_features(X, self.groupby)
        return X

    def fit(self, X, y=None):
        X = self.__validate_input(X)
        global_mu = X.drop(columns=[self.groupby]).mean()
        self._set_factors(
            (X.groupby(self.groupby).mean() / global_mu).to_dict())
        return self

    def transform(self, X, y=None):
        "Removes the factors influences."
        X = self.__validate_input(X)
        for (col, factors_dict) in self.factors.items():
            X[col] = X[col] / X[self.groupby].map(factors_dict)
        return X

    def inverse_transform(self, X, y=None):
        "Reinstates the factors influences."
        X = self.__validate_input(X)
        for (col, factors_dict) in self.factors.items():
            X[col] = X[col] * X[self.groupby].map(factors_dict)
        return X

    def save(self, file_path: str):
        joblib.dump(self, file_path)

    @staticmethod
    def load(file_path: str):
        return joblib.load(file_path)
