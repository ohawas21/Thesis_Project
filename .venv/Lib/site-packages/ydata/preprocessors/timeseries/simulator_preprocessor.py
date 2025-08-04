from typing import List, Union

from numpy import array, concatenate, zeros
from pandas import DataFrame, concat
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ydata.__serializer import SerializerMixin
from ydata.preprocessors.timeseries.utils import LambertWTransformer


class TimeSeriesSimulatorPreprocessor(BaseEstimator, TransformerMixin, SerializerMixin):
    def __init__(
        self,
        *,
        num_cols: List[str] = None,
        cat_cols: List[str] = None,
        dt_cols: List[str] = None
    ):

        assert (
            cat_cols is None
        ), "Time Series simulator model does not support categorical variables."
        self.num_cols = [] if num_cols is None else num_cols
        self.dt_cols = [] if dt_cols is None else dt_cols

        self.num_pipeline = make_pipeline(
            StandardScaler(),
            LambertWTransformer(),
            StandardScaler(),
        )

    def fit(self, X: Union[DataFrame, array]):
        self.num_pipeline.fit(X[self.num_cols]) if self.num_cols else DataFrame(
            index=X.index, data=zeros([len(X), 0])
        )
        return self

    def transform(self, X: Union[DataFrame, array]):
        num_data = (
            self.num_pipeline.transform(X[self.num_cols])
            if self.num_cols
            else DataFrame(index=X.index, data=zeros([len(X), 0]))
        )
        dt_data = X[self.dt_cols].values

        if dt_data.shape[1] > 0:
            assert (
                dt_data.shape[1] == num_data.shape[1]
            ), "Something went wrong while applying the Inverse of Lambert to the provided data. Please verify your inputs."
            data = concatenate([dt_data, num_data], axis=1)
        else:
            data = num_data
        return DataFrame(data, columns=self.dt_cols + self.num_cols)

    def inverse_transform(self, X):
        num_data = X[self.num_cols].values
        num_data = (
            self.num_pipeline.inverse_transform(num_data)
            if self.num_cols
            else DataFrame(index=X.index, data=zeros([len(X), 0]))
        )

        if len(self.dt_cols):
            dt_data = X[self.dt_cols].values
            data = concat([dt_data, num_data], axis=1)
        else:
            data = num_data
        return DataFrame(data, columns=self.dt_cols + self.num_cols)
