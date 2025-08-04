"""Predictive model.

Reference: J. Jordon, J. Yoon, M. van der Schaar,
           "Measuring the quality of Synthetic data for use in competitions,"
           KDD Workshop on Machine Learning for Medicine and Healthcare, 2018
Paper Link: https://arxiv.org/abs/1806.11345
Contact: jsyoon0823@gmail.com
"""

from enum import Enum
from typing import List

from numpy import abs as np_abs
from numpy import argmax
from numpy import array as np_array
from numpy import maximum as np_maximum
from numpy import mean as np_mean
from pandas import get_dummies
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, mean_absolute_error, mean_squared_error,
                             roc_auc_score)
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


def mean_absolute_percentage_error(y_true, y_pred):
    """Mean absolute percentage error regression loss.

    Examples
    --------
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> mean_absolute_percentage_error(y_true, y_pred) # doctest: +ELLIPSIS
    0.3273...

    >>> y_true = [[0.5, 1], [-1, 1], [7, -6]]
    >>> y_pred = [[0, 2], [-1, 2], [8, -5]]
    >>> mean_absolute_percentage_error(y_true, y_pred) # doctest: +ELLIPSIS
    0.5515...
    """
    eps = 1e-16
    y_t, y_p = np_array(y_true), np_array(y_pred)
    return np_mean(np_abs(y_p - y_t) / np_maximum(np_abs(y_t), eps))


class Task(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class _Model:
    def __init__(
        self, name: str, model_class, task: Task = Task.CLASSIFICATION, **parameters
    ) -> None:
        self.name = name
        self.model_class = model_class
        self._model = model_class(**parameters)
        self.task = task
        self.parameters = parameters

    def fit(self, X, y):
        self._model = self.model_class(**self.parameters)
        return self._model.fit(X, y)

    def predict_proba(self, test_x):
        if self.task == Task.CLASSIFICATION:
            return self._model.predict_proba(test_x)
        else:
            return self._model.predict(test_x)

    def __str__(self) -> str:
        return f"Model(name={self.name}, task={self.task})"

    def __repr__(self) -> str:
        return str(self)


class ModelsList(Enum):
    # regression models
    LINEAR = _Model("LINEAR", LinearRegression, Task.REGRESSION)
    MLP = _Model(
        "MLP", MLPRegressor, Task.REGRESSION, hidden_layer_sizes=(50, 50), max_iter=500
    )
    TREE_R = _Model("TREE_R", DecisionTreeRegressor, Task.REGRESSION)
    RIDGE = _Model("RIDGE", Ridge, Task.REGRESSION)
    LASSO = _Model("LASSO", Lasso, Task.REGRESSION)
    SVR = _Model("SVR", SVR, Task.REGRESSION)
    # classifications models
    NN = _Model("NN", MLPClassifier, hidden_layer_sizes=(200, 200))
    RANDOMFOREST = _Model("RANDOMFOREST", RandomForestClassifier)
    GAUSSIANNB = _Model("GAUSSIANNB", GaussianNB)
    BERNOULLINB = _Model("BERNOULLINB", BernoulliNB)
    GBM = _Model("GBM", GradientBoostingClassifier)
    EXTRA_TREES = _Model("EXTRA_TREES", ExtraTreesClassifier, n_estimators=20)
    LDA = _Model("LDA", LinearDiscriminantAnalysis)
    ADABOOST = _Model("ADABOOST", AdaBoostClassifier)
    BAGGING = _Model("BAGGING", BaggingClassifier)

    @classmethod
    def elements(cls) -> List[str]:
        return [e.name for e in cls]

    @classmethod
    def elements_by_task(cls, task: Task) -> List[str]:
        return [e.name for e in cls if e.value.task == task]


class Score:
    def __init__(self, name, function, **parameters) -> None:
        self.name = name
        self.function = function
        self.parameters = parameters

    def score(self, *args):
        return self.function(*args, **self.parameters)

    def __str__(self) -> str:
        return f"Score(name={self.name})"

    def __repr__(self) -> str:
        return str(self)


def _roc_auc_score(y_true, y_score, **kwargs):
    return roc_auc_score(get_dummies(y_true), y_score, **kwargs)


def _accuracy(y_true, y_hat):
    labels = argmax(y_hat, axis=1)
    return accuracy_score(y_true, labels)


class ScoreList(Enum):
    AUC = Score("AUC", _roc_auc_score, multi_class="ovr", average="weighted")
    APR = Score("APR", average_precision_score)
    MAE = Score("MAE", mean_absolute_error)
    MAPE = Score("MAPE", mean_absolute_percentage_error)
    RMSE = Score("RMSE", mean_squared_error, squared=False)
    F1 = Score("F1", f1_score)
    ACCURACY = Score("ACCURACY", _accuracy)

    @classmethod
    def elements(cls) -> List[str]:
        return [e.name for e in cls]


class PredictiveModel:
    def __init__(self, model_name: str, task: Task) -> None:
        self._model = self._get_model(model_name, task)

    def _get_model(self, model_name: str, task: Task) -> _Model:
        model_name = model_name.upper()
        available_models = ModelsList.elements_by_task(task)
        assert (
            model_name in available_models
        ), f"Model [{model_name}] not supported for {task}."
        return ModelsList[model_name].value

    def _get_scorer(self, metric_name: str) -> Score:
        metric_name = metric_name.upper()
        assert metric_name in ScoreList.elements(
        ), f"Invalid metric [{metric_name}]"
        return ScoreList[metric_name].value

    def evaluate(self, train_x, train_y, test_x, test_y, metric_name: str):
        """Evaluate predictive model performance.

        Args:
        - y_true: original testing labels
        - y_hat: prediction on testing data
        - metric_name: selected metric

        Returns:
        - score: performance of the predictive model
        """
        metric = self._get_scorer(metric_name)
        self._model.fit(train_x, train_y)
        y_hat = self._model.predict_proba(test_x)
        return metric.score(test_y, y_hat)
