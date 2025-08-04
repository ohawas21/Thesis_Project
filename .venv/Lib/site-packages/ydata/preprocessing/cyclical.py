"""Utilities for cyclical features."""
from typing import List, Optional

from numpy import arccos, array, cos, pi, shape, sin
from sklearn.base import BaseEstimator, TransformerMixin


class CyclicalFeatures(BaseEstimator, TransformerMixin):
    """Apply sin+cos transformation for features with a natural cyclical nature
    (e.g. hour of day)."""

    def __init__(self, maxs: Optional[List[float]] = None, drop=True):
        """Transformer to convert numerical cyclical features (sin + cos).

        Args:
            maxs (List[float], optional): list of maximum feature values of columns.
            drop (bool): drop original columns after transformation
        """
        self.maxs = maxs
        self.drop = drop
        self.cols = ()

    @staticmethod
    def sin(x, max):
        return sin(2 * pi * x / max)

    @staticmethod
    def cos(x, max):
        return cos(2 * pi * x / max)

    @staticmethod
    def _invert_cos_sin(cos, sin):
        if (
            sin >= 0
        ):  # Arccos is better than arcsin since it returns always positive rads (following method is simpler)
            return arccos(cos)
        else:
            return 2 * pi - arccos(cos)

    @staticmethod
    def _rads_to_val(rads, max):
        return rads / (2 * pi) * max

    def fit(self, X, y=None):
        self.maxs = array(X.max()) if self.maxs is None else self.maxs
        assert shape(X)[1] == len(self.maxs), (
            f"The length of 'maxs' ({len(self.maxs)}) should match the "
            f"column dimensionality of X ({shape(X)[1]})"
        )
        return self

    def transform(self, X, y=None):
        X = X.copy()
        self.cols = X.columns
        for i, col in enumerate(self.cols):
            X[col + "_sin"] = self.sin(X[col], self.maxs[i])
            X[col + "_cos"] = self.cos(X[col], self.maxs[i])
        if self.drop:
            X = X.drop(columns=self.cols)
        return X

    def inverse_transform(self, X, y=None):
        # TODO: Add checks for input validation
        X = X.copy()
        for i, col in enumerate(self.cols):
            max = self.maxs[i]
            # Converts sin/cos values to radians
            rads = X[[f"{col}_cos", f"{col}_sin"]].apply(
                lambda x: self._invert_cos_sin(x[0], x[1]), axis=1
            )
            # Restores from  radians to original range
            X.loc[:, col] = rads.apply(lambda x: self._rads_to_val(x, max))
            if self.drop:
                X = X.drop(columns=[f"{col}_sin", f"{col}_cos"])
            return X
