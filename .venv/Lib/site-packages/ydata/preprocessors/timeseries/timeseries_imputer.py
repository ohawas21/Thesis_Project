import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error


def forward_fill_strategy(df: pd.DataFrame):
    return df.ffill().bfill()


def backward_fill_strategy(df: pd.DataFrame):
    return df.bfill().ffill()


def linear_interpolation_strategy(df: pd.DataFrame):
    return df.interpolate()


def spline_interpolation_strategy(df: pd.DataFrame):
    return df.interpolate(method="spline", order=3)


def knn_mean_strategy(df: pd.DataFrame):
    def knn_mean(ts, n):
        for i, val in enumerate(ts):
            if np.isnan(val):
                n_by_2 = np.ceil(n / 2)
                lower = np.max([0, int(i - n_by_2)])
                upper = np.min([len(ts) + 1, int(i + n_by_2)])
                ts_near = np.concatenate([ts[lower:i], ts[i:upper]])
                ts.iloc[i] = np.nanmean(ts_near)
        return ts

    return knn_mean(df, 8)


def seasonal_mean_strategy(df: pd.DataFrame):
    def seasonal_mean(ts, n, lr=0.7):
        """
        Compute the mean of corresponding seasonal periods
        ts: 1D array-like of the time series
        n: Seasonal window length of the time series
        """
        for i, val in enumerate(ts):
            if np.isnan(val):
                ts_seas = ts[i - 1:: -n]  # previous seasons only
                if np.isnan(np.nanmean(ts_seas)):
                    ts_seas = np.concatenate(
                        [ts[i - 1:: -n], ts[i::n]]
                    )  # previous and forward
                ts.iloc[i] = np.nanmean(ts_seas) * lr
        return ts

    return seasonal_mean(df, n=12, lr=1.25)


def calc_strategy_error(df: pd.DataFrame, df_implaced: pd.DataFrame):
    return np.round(mean_squared_error(df, df_implaced), 2)


class TimeSeriesImputer(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_impute):
        self.cols_to_impute = cols_to_impute
        self.best_imputer = {}

    def fit(self, X):
        for column in self.cols_to_impute:
            self.best_imputer[column] = self.fit_column(
                X[column].astype(float))
        return self

    def fit_column(self, X):
        X_filtered = X.copy()
        X_filtered.dropna(inplace=True)
        # create df with missing values for evaluation
        remove_n = int(X_filtered.shape[0] / 10)  # todo: error prune
        drop_indices = np.random.choice(
            X_filtered.index, remove_n, replace=False)
        X_impute = X_filtered.copy()
        X_impute.loc[drop_indices] = np.nan

        imputers = [backward_fill_strategy, forward_fill_strategy]
        # linear_interpolation_strategy]
        # spline_interpolation_strategy, knn_mean_strategy, seasonal_mean_strategy]
        errors = []
        for i in range(len(imputers)):
            errors.append(calc_strategy_error(
                X_filtered, imputers[i](X_impute.copy())))
        return imputers[errors.index(min(errors))]

    def transform(self, X, y=None):
        for column in self.cols_to_impute:
            X.loc[:, column] = self.best_imputer[column](X[column])
        return X

    def inverse_transform(self, X):
        return X
