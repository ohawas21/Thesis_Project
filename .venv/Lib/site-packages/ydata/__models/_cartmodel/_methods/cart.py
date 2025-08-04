"""File to define the Cart Method logic CART synthesis
"""
import math
from dataclasses import asdict, dataclass
from random import sample
from typing import Dict, List

from numpy import array
from numpy import max as npmax
from numpy import min as npmin
from numpy import nan, where, zeros
from numpy.random import default_rng
from pandas import DataFrame
from pandas import DataFrame as pdDataFrame
from pandas import cut
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor

from ydata.__models._cartmodel._methods import BaseMethod
from ydata.__models._cartmodel._methods.utils import proper, smooth
from ydata.utils.data_types import CATEGORICAL_DTYPES, DataType, VariableType
from ydata.utils.random import RandomSeed


@dataclass
class CartHyperparameterConfig:
    min_samples_leaf: int = 2
    max_depth: int = None
    splitter: str = "random"


class CartHyperparameterStrategy(object):
    def __call__(self, X, y, dtypes, y_dtype) -> CartHyperparameterConfig:
        # Functor used as dispatcher
        # For now we use only a naive strategy, but we can imagine more advanced strategies based
        # on datatypes, shapes or column number.
        return self.__naive(X, y, dtypes, y_dtype)

    @staticmethod
    def __naive(X, y, dtypes, y_dtype) -> CartHyperparameterConfig:
        LOWER = 5  # Lower bounds from which there is no constraint
        N = X.shape[0]
        r = int(math.log10(N - 10**LOWER)) if (N > 10**LOWER) else 0
        config = CartHyperparameterConfig(
            max_depth=None if r == 0 else max(20, 50 - (10 * r))
        )
        return config


class CARTMethod(BaseMethod):
    def __init__(
        self,
        y_dtype,
        smoothing=False,
        proper=False,
        minibucket=5,
        random_state: RandomSeed = None,
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
        self.minibucket = minibucket
        self.random_state = random_state
        self.cart = None
        self.hyperparam_strat = CartHyperparameterStrategy()

    def __init_cart(self, X, y, dtypes, y_dtype, min_sample_leaf, random_state):
        params = self.hyperparam_strat(X, y, dtypes, y_dtype)
        return DecisionTreeRegressor(random_state=random_state, **asdict(params))

    def fit(self, X, y, dtypes: Dict = None):
        self.cart = self.__init_cart(
            X, y, dtypes, self.y_dtype, self.minibucket, self.random_state
        )

        if self.proper:
            X, y = proper(X=X, y=y, random_state=self.random_state)

        X, y = self.prepare_dfs(X=X, y=y, dtypes=dtypes,
                                normalise_num_cols=False)
        if self.y_dtype in [DataType.NUMERICAL, DataType.DATE]:
            self.y_real_min, self.y_real_max = npmin(y), npmax(y)

        X = X.to_numpy()
        y = y.to_numpy()
        self.cart.fit(X, y)

        # save the y distribution wrt trained tree nodes
        leaves = self.cart.apply(X)
        leaves_y_df = pdDataFrame({"leaves": leaves, "y": y})
        self.leaves_y_dict = (
            leaves_y_df.groupby("leaves").apply(
                lambda x: x.to_numpy()[:, -1]).to_dict()
        )

    def predict(self, X, dtypes: Dict = None, random_state: RandomSeed = None):
        rng = default_rng(seed=random_state)
        X_test_df, _ = self.prepare_dfs(
            X=X, dtypes=dtypes, normalise_num_cols=False, fit=False
        )
        # predict the leaves and for each leaf randomly sample from the observed values
        X_test = X_test_df.to_numpy()
        leaves_pred = self.cart.apply(X_test)
        y_pred = zeros(len(leaves_pred), dtype=object)

        leaves_pred_index_df = pdDataFrame(
            {"leaves_pred": leaves_pred, "index": range(len(leaves_pred))}
        )
        leaves_pred_index_dict = (
            leaves_pred_index_df.groupby("leaves_pred")
            .apply(lambda x: x.to_numpy()[:, -1])
            .to_dict()
        )
        for leaf, indices in leaves_pred_index_dict.items():
            y_pred[indices] = rng.choice(
                self.leaves_y_dict[leaf], size=len(indices), replace=True
            )

        if self.smoothing and self.y_dtype == DataType.NUMERICAL:
            y_pred = smooth(self.y_dtype, y_pred,
                            self.y_real_min, self.y_real_max)

        return y_pred


class SeqCARTMethod(BaseMethod):
    def __init__(
        self,
        y_dtype: DataType,
        smoothing: bool = False,
        proper: bool = False,
        minibucket: int = 5,
        random_state: RandomSeed = None,
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
        self.minibucket = minibucket
        self.random_state = random_state
        self.cluster = kwargs.get("cluster")
        if self.cluster is not None:
            self.cluster = array(self.cluster)
        self.order = kwargs.get("order")
        self.cart = None
        self.hyperparam_strat = CartHyperparameterStrategy()
        self.dummy = None

    def __init_cart(self, X, y, dtypes, y_dtype, min_sample_leaf, random_state):
        params = self.hyperparam_strat(X, y, dtypes, y_dtype)
        return DecisionTreeRegressor(random_state=random_state, **asdict(params))

    def fit(
        self, X, y, autoregressive: bool = False, dtypes: Dict = None, *args, **kwargs
    ):

        self.cart = self.__init_cart(
            X, y, dtypes, self.y_dtype, self.minibucket, self.random_state
        )

        if self.proper:
            X, y = proper(X=X, y=y, random_state=self.random_state)

        if autoregressive:
            dtypes = dict(dtypes)
            if self.y_dtype["datatype"] == DataType.NUMERICAL:
                X["dummy"] = cut(y, bins=50, labels=["a" + str(i)
                                 for i in range(50)])
                X["dummy"] = LabelEncoder().fit_transform(
                    X["dummy"]).astype(int) + 1
                self.dummy = X["dummy"].copy()
                autoregressive = ["dummy"]
                dtypes["dummy"] = {
                    "datatype": DataType.CATEGORICAL,
                    "vartype": VariableType.INT,
                }
        else:
            autoregressive = None
        X, y = self.prepare_dfs(X=X, y=y, dtypes=dtypes,
                                normalise_num_cols=True)

        assert self.cluster is not None, "conditioning_col is not a valid one"

        if self.y_dtype["datatype"] == DataType.NUMERICAL:
            self.y_real_min, self.y_real_max = npmin(y), npmax(y)

        # This is required as the new CategoryDtype does not support median
        X = X.astype({col: 'int' for col in X.select_dtypes(include=['category']).columns})
        X = X.fillna(X.median())
        X = X.to_numpy()

        if self.y_dtype["datatype"] in CATEGORICAL_DTYPES:
            if "NaN" in y.name:
                y = y.to_numpy().astype("int")
            else:
                y = y.to_numpy().astype(str).astype("O")
        else:
            y = y.to_numpy()

        self.cart.fit(X, y)

        # save the y distribution wrt trained tree nodes
        leaves = self.cart.apply(X)
        leaves_y_df = DataFrame(
            {"leaves": leaves, "cluster": self.cluster, "y": y})
        self.leaves_y_dict = (
            leaves_y_df.groupby(["leaves", "cluster"])
            .apply(lambda x: x.to_numpy()[:, -1])
            .to_dict()
        )

    def _predict(self, leaves_pred: List, cluster: List, random_state: RandomSeed = None):
        rng = default_rng(seed=random_state)
        y_pred = zeros(len(leaves_pred), dtype=object)
        leaves_pred_index_df = DataFrame(
            {
                "leaves_pred": leaves_pred,
                "cluster": cluster[: len(leaves_pred)],
                "index": range(len(leaves_pred)),
            }
        )
        leaves_pred_index_dict = (
            leaves_pred_index_df.groupby(["leaves_pred", "cluster"])
            .apply(lambda x: x.to_numpy()[:, -1])
            .to_dict()
        )
        for leaf, indices in leaves_pred_index_dict.items():
            if (leaf[0], leaf[1]) in self.leaves_y_dict:
                indices = list(indices.astype("int"))
                y_pred[indices] = rng.choice(
                    self.leaves_y_dict[(leaf[0], leaf[1])],
                    size=len(indices),
                    replace=True,
                )
            else:
                ll = sample(list(self.leaves_y_dict.keys()), 1)[0]
                indices = list(indices.astype("int"))
                y_pred[indices] = rng.choice(
                    self.leaves_y_dict[ll], size=len(indices), replace=True
                )
        if self.y_dtype["datatype"] in CATEGORICAL_DTYPES:
            y_pred = where(y_pred == "NA", nan, y_pred)
        return y_pred

    def predict(
        self, X, autoregressive: bool, dtypes: Dict = None, cluster: List = None, random_state: RandomSeed = None
    ):
        if autoregressive and self.dummy is not None:
            dtypes = dict(dtypes)
            X["dummy"] = self.dummy.iloc[: len(X)].copy().tolist()
            autoregressive = ["dummy"]
            dtypes["dummy"] = {
                "datatype": DataType.CATEGORICAL,
                "vartype": VariableType.INT,
            }
        else:
            autoregressive = None
        X_test, _ = self.prepare_dfs(
            X=X, dtypes=dtypes, normalise_num_cols=True, fit=False
        )
        X_test = X_test.fillna(X_test.median())
        X_test = X_test.to_numpy()

        leaves_pred = self.cart.apply(X_test)
        y_pred = self._predict(leaves_pred, cluster, random_state=random_state)

        if self.smoothing and self.y_dtype["datatype"] == DataType.NUMERICAL:
            y_pred = smooth(
                self.y_dtype["datatype"], y_pred, self.y_real_min, self.y_real_max
            )
        return y_pred
