from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose

from ydata.__serializer import SerializerMixin
from ydata.preprocessors.preprocess_methods import (CategoricalEncoder, CategoricalImputer, DateTimeImputer,
                                                    DateTimeTransformer, IdentityTransformer, IntegerTransformer)
from ydata.preprocessors.timeseries.timeseries_imputer import TimeSeriesImputer


class TimeSeriesIdentity(BaseEstimator, TransformerMixin):
    def fit(self, X):
        return self

    def transform(self, X, y=None):
        return X

    def inverse_transform(self, X):
        return X


class TimeSeriesEquidistance(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.mode = (pd.Series(X.index) -
                     pd.Series(X.index).shift(1)).mode().iloc[0]
        return self

    def transform(self, X, y=None):
        return X.resample(self.mode).asfreq()

    def inverse_transform(self, X):
        return X


class TimeSeriesOutlierDetector(BaseEstimator, TransformerMixin):
    def __init__(self, *, outlier_frac: float = 0.01):
        super().__init__(novelty=True)
        self.outlier_frac = outlier_frac

    def fit(self, X):
        self.model = IsolationForest(contamination=self.outlier_frac)
        self.model.fit(X)
        return self

    def transform(self, X, y=None):
        outliers_map = self.model.predict(X)
        outliers_idx = np.where(outliers_map == -1)
        return X.drop(X.index[outliers_idx])

    def inverse_transform(self, X):
        return X


class TimeSeriesDetrend(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_detrend):
        self.cols_to_detrend = cols_to_detrend
        self.trends = {}

    def fit(self, X, y=None):
        for column in self.cols_to_detrend:
            self.trends[column] = self.fit_column(X[column].astype(float))
        return self

    def fit_column(self, X):
        trend = seasonal_decompose(
            X, model="addicative", extrapolate_trend="freq", period=1
        )
        return trend

    def transform(self, X, y=None):
        for column in self.cols_to_detrend:
            X[column] = X[column] - self.trends[column].trend
        return X

    def inverse_transform(self, X):
        for column in self.cols_to_detrend:
            X[column] = X[column] + self.trends[column].trend
        return X


class TimeSeriesMinMaxScaler(MinMaxScaler):
    def transform(self, X, y=None):
        X = pd.DataFrame(data=super().transform(
            X), columns=X.columns, index=X.index)
        return X

    def inverse_transform(self, X):
        X = pd.DataFrame(
            data=super().inverse_transform(X), columns=X.columns, index=X.index
        )
        return X


class TimeSeriesIntegerTransformer(IntegerTransformer):
    def transform(self, X, y=None):
        X = pd.DataFrame(data=super().transform(
            X), columns=X.columns, index=X.index)
        return X

    def inverse_transform(self, X):
        X = pd.DataFrame(
            data=super().inverse_transform(X.to_numpy()),
            columns=X.columns,
            index=X.index,
        )
        return X


class TimeSeriesIdentityTransformer(IdentityTransformer):
    def transform(self, X, y=None):
        X = pd.DataFrame(data=super().transform(
            X), columns=X.columns, index=X.index)
        return X

    def inverse_transform(self, X):
        X = pd.DataFrame(
            data=super().inverse_transform(X.to_numpy()),
            columns=X.columns,
            index=X.index,
        )
        return X


class TimeSeriesCategoricalImputer(CategoricalImputer):
    def transform(self, X, y=None):
        X = pd.DataFrame(data=super().transform(
            X), columns=X.columns, index=X.index)
        return X

    def inverse_transform(self, X):
        X = pd.DataFrame(
            data=super().inverse_transform(X.to_numpy()),
            columns=X.columns,
            index=X.index,
        )
        return X


class TimeSeriesCategoricalEncoder(CategoricalEncoder):
    def transform(self, X, y=None):
        X = pd.DataFrame(data=super().transform(
            X), columns=X.columns, index=X.index)
        return X

    def inverse_transform(self, X):
        X = pd.DataFrame(
            data=super().inverse_transform(X.to_numpy()),
            columns=X.columns,
            index=X.index,
        )
        return X


class TimeSeriesDateTimeImputer(DateTimeImputer):
    def transform(self, X, y=None):
        X = pd.DataFrame(data=super().transform(
            X), columns=X.columns, index=X.index)
        return X

    def inverse_transform(self, X):
        X = pd.DataFrame(
            data=super().inverse_transform(X), columns=X.columns, index=X.index
        )
        return X


class TimeSeriesDateTimeTransformer(DateTimeTransformer):
    def transform(self, X, y=None):
        X = pd.DataFrame(data=super().transform(
            X), columns=X.columns, index=X.index)
        return X

    def inverse_transform(self, X):
        X = pd.DataFrame(
            data=super().inverse_transform(X.to_numpy()),
            columns=X.columns,
            index=X.index,
        )
        for col_name in X.columns.to_list():
            X[col_name] = pd.to_datetime(X[col_name])
        return X


class TimeSeriesPreprocessor(BaseEstimator, TransformerMixin, SerializerMixin):
    def __init__(
        self,
        *,
        num_cols: List[str] = None,
        cat_cols: List[str] = None,
        dt_cols: List[str] = None,
        filter_outliers: bool = False
    ):
        self.num_cols = [] if num_cols is None else num_cols
        self.cat_cols = [] if cat_cols is None else cat_cols
        self.dt_cols = [] if dt_cols is None else dt_cols

        self.num_pipeline = make_pipeline(
            TimeSeriesImputer(self.num_cols),
            TimeSeriesMinMaxScaler(),
        )

        self.pre_pipeline = make_pipeline(TimeSeriesEquidistance())

        self.cat_pipeline = make_pipeline(
            TimeSeriesCategoricalImputer(),
            TimeSeriesCategoricalEncoder(),
        )

        self.dt_pipeline = make_pipeline(
            TimeSeriesDateTimeImputer(),
            TimeSeriesDateTimeTransformer(),
            TimeSeriesMinMaxScaler(),
        )

    def fit(self, X, **kwargs):
        self.pre_pipeline.fit(X)
        self.num_pipeline.fit(X[self.num_cols]) if self.num_cols else pd.DataFrame(
            index=X.index, data=np.zeros([len(X), 0])
        )
        self.cat_pipeline.fit(X[self.cat_cols]) if self.cat_cols else pd.DataFrame(
            index=X.index, data=np.zeros([len(X), 0])
        )
        self.dt_pipeline.fit(X[self.dt_cols]) if self.dt_cols else pd.DataFrame(
            index=X.index, data=np.zeros([len(X), 0])
        )
        return self

    def transform(self, X):
        X = self.pre_pipeline.transform(X)
        num_data = (
            self.num_pipeline.transform(X[self.num_cols])
            if self.num_cols
            else pd.DataFrame(index=X.index, data=np.zeros([len(X), 0]))
        )
        cat_data = (
            self.cat_pipeline.transform(X[self.cat_cols])
            if self.cat_cols
            else pd.DataFrame(index=X.index, data=np.zeros([len(X), 0]))
        )
        dt_data = (
            self.dt_pipeline.transform(X[self.dt_cols])
            if self.dt_cols
            else pd.DataFrame(index=X.index, data=np.zeros([len(X), 0]))
        )
        return pd.concat([cat_data, num_data, dt_data], axis=1)

    def inverse_transform(self, X):
        num_data = X[self.num_cols]

        cat_cols = [
            self.cat_pipeline.steps[-1][1].old_cols_to_new[col_name]
            for col_name in self.cat_cols
        ]
        cat_data = X[sum(cat_cols, [])]
        dt_data = X[self.dt_cols]
        num_data = (
            self.num_pipeline.inverse_transform(num_data)
            if self.num_cols
            else pd.DataFrame(index=X.index, data=np.zeros([len(X), 0]))
        )
        cat_data = (
            self.cat_pipeline.inverse_transform(cat_data)
            if self.cat_cols
            else pd.DataFrame(index=X.index, data=np.zeros([len(X), 0]))
        )
        dt_data = (
            self.dt_pipeline.inverse_transform(dt_data)
            if self.dt_cols
            else pd.DataFrame(index=X.index, data=np.zeros([len(X), 0]))
        )

        result = pd.concat([num_data, cat_data, dt_data], axis=1)
        return result
