"""A method adapted for TargetEncoding.

As target encodding tends to suffer from overfitting, it is recommended
to follow a stratefied strategy in order it
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold


class AvgTargetEncoder(BaseEstimator, TransformerMixin):
    # The original idea originally taken (by all) from it is lit:
    # https://www.kaggle.com/brandenkmurray/two-sigma-connect-rental-listing-inquiries/it-is-lit

    def __init__(self, variable, target, fold=5, params=None):
        self.variable = variable
        self.target = target
        self.hcc_name = "_".join(["hcc", self.variable, self.target])
        self.params = params if params is not None else {
            "f": 1, "g": 1, "k": 5}

        self.fold = StratifiedKFold(fold)

    def fit(self, X, y=None):
        self.y = y
        self.dummy = pd.get_dummies(y).astype(int)
        return self

    def transform(self, X, y=None):
        prior = self.dummy.mean()[self.target]

        train_mask = X.train == 1
        X_train = X[train_mask]
        X_test = X[~train_mask]
        self.y = self.y[train_mask]
        self.dummy = self.dummy[train_mask]

        X_train = X_train.join(self.dummy)

        encoding = AvgTargetEncoder.encode(
            X_train, prior, self.variable, self.target, self.hcc_name, self.params
        )
        test_df = (
            X_test[[self.variable]]
            .join(encoding, on=self.variable, how="left")[self.hcc_name]
            .fillna(prior)
        )

        dfs = []
        for train, test in self.fold.split(np.zeros(len(X_train)), self.y):
            train_split = X_train.iloc[train]
            test_split = X_train.iloc[test]
            encoding = AvgTargetEncoder.encode(
                train_split,
                prior,
                self.variable,
                self.target,
                self.hcc_name,
                self.params,
            )
            df = (
                test_split[[self.variable]]
                .join(encoding, on=self.variable, how="left")[self.hcc_name]
                .fillna(prior)
            )
            dfs.append(df)
        dfs.append(test_df)
        df = pd.concat(dfs)
        df = df.reindex(X.index)

        return df.to_frame(name=self.hcc_name)

    def get_feature_names(self):
        return [self.hcc_name]

    @staticmethod
    def encode(X_train, prior, variable, target, hcc_name, params):
        f, g, k = params["f"], params["g"], params["k"]
        grouped = X_train.groupby(variable)[target].agg(
            {"size": "size", "mean": "mean"}
        )
        grouped["lambda"] = 1 / (g + np.exp((k - grouped["size"]) / f))
        grouped[hcc_name] = (
            grouped["lambda"] * grouped["mean"] +
            (1 - grouped["lambda"]) * prior
        )
        return grouped
