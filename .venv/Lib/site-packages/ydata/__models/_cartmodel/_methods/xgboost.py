from typing import Dict

from numpy import concatenate, max, min, random, reshape, zeros
from pandas import DataFrame
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from ydata.__models._cartmodel._methods import BaseMethod
from ydata.__models._cartmodel._methods.utils import proper, smooth
from ydata.utils.data_types import CATEGORICAL_DTYPES, DataType


class XGBMethod(BaseMethod):
    def __init__(
        self,
        dtype: DataType,
        smoothing: bool = False,
        proper: bool = False,
        minibucket: int = 5,
        random_state: int = None,
        *args,
        **kwargs
    ):
        self.dtype = dtype
        self.smoothing = smoothing
        self.proper = proper
        self.minibucket = minibucket
        self.random_state = random_state
        self.length = kwargs.get("length")
        self.cluster = kwargs.get("cluster")
        self.one_hot_cat_cols = kwargs.get("one_hot_cat_cols")
        self.order = kwargs.get("order")

        if self.dtype in CATEGORICAL_DTYPES:
            self.xgboost = GradientBoostingClassifier()
        if self.dtype == DataType.NUMERICAL:
            self.xgboost = GradientBoostingRegressor()

    def fit(self, X, y, autoregressive: Dict, *args, **kwargs):
        if self.proper:
            X, y = proper(X=X, y=y, random_state=self.random_state)

        X, y = self.prepare_dfs(
            X=X,
            y=y,
            normalise_num_cols=True,
            one_hot_cat_cols=self.one_hot_cat_cols,
            autoregressive=autoregressive,
        )

        if self.dtype == DataType.NUMERICAL:
            self.y_real_min, self.y_real_max = min(y), max(y)

        X = X.fillna(X.median())

        X = X.to_numpy()
        y = y.to_numpy()
        self.xgboost.fit(X, y)

        leaves = self.xgboost.apply(X)
        yy = concatenate([y] * self.xgboost.n_estimators)

        if len(leaves.shape) == 2:
            ll = reshape(leaves, (leaves.shape[0] * leaves.shape[1]))
        else:
            ll = reshape(leaves, (leaves.shape[0] * leaves.shape[1], leaves.shape[2]))[
                :, 0
            ]

        leaves_y_df = DataFrame({"leaves": ll, "y": yy})
        self.leaves_y_dict = (
            leaves_y_df.groupby("leaves").apply(
                lambda x: x.to_numpy()[:, -1]).to_dict()
        )

    def predict(self, X_test, autoregressive: Dict, *args, **kwargs):
        X_test, _ = self.prepare_dfs(
            X=X_test,
            normalise_num_cols=True,
            one_hot_cat_cols=self.one_hot_cat_cols,
            fit=False,
            autoregressive=autoregressive,
        )

        X_test = X_test.to_numpy()
        leaves_pred = self.xgboost.apply(X_test)
        y_pred = zeros(len(leaves_pred), dtype=object)
        ll_ = self.xgboost.n_estimators * len(leaves_pred)
        if len(leaves_pred.shape) == 2:
            ll = reshape(
                leaves_pred, (leaves_pred.shape[0] * leaves_pred.shape[1]))
        else:
            ll = reshape(
                leaves_pred,
                (leaves_pred.shape[0] *
                 leaves_pred.shape[1], leaves_pred.shape[2]),
            )[:, 0]

        leaves_pred_index_df = DataFrame(
            {"leaves_pred": ll, "index": range(ll_)})
        leaves_pred_index_dict = (
            leaves_pred_index_df.groupby("leaves_pred")
            .apply(lambda x: x.to_numpy()[:, -1])
            .to_dict()
        )
        for leaf, indices in leaves_pred_index_dict.items():
            indices = [(x // self.xgboost.n_estimators).astype("int")
                       for x in indices]
            indices = list(dict.fromkeys(indices))
            y_pred[indices] = random.choice(
                self.leaves_y_dict[leaf], size=len(indices), replace=True
            )

        if self.smoothing and self.dtype == DataType.NUMERICAL:
            y_pred = smooth(self.dtype, y_pred,
                            self.y_real_min, self.y_real_max)

        return y_pred
