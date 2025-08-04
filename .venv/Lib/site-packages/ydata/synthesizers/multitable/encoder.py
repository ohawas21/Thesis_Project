from numpy import nan_to_num
from pandas import DataFrame as pdDataFrame
from sklearn.cluster import AffinityPropagation, Birch, MeanShift
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from ydata.metadata import Metadata
from ydata.metadata.multimetadata import TableSchema
# from ydata.datascience.common import EncoderType
from ydata.synthesizers.multitable import EncoderType


class BaseEncoder:
    def __init__(self) -> None:
        self._encoded_columns = []
        self._max_features = None

    @property
    def encoded_columns(self) -> list[str]:
        return self._encoded_columns

    def _preprocess(
        self,
        X: pdDataFrame,
        metadata: Metadata,
        schema: TableSchema,
        table: str = ""
    ):
        keys = schema.get_keys()
        uniques = set(w.column for w in metadata.warnings.get("unique", []))
        self.invalid_columns = set(keys) | uniques

        num_cols = [
            f"{table}.{c}" if table else c
            for c in metadata.numerical_vars
        ]
        cat_cols = [
            f"{table}.{c}" if table else c
            for c in metadata.categorical_vars
        ]
        # TODO add preprocessing to date vars
        # metadata.date_vars

        if table:
            self.invalid_columns = set(
                f"{table}.{col}"
                for col in self.invalid_columns
            )

        feat_columns = num_cols + cat_cols
        if all([c in self.invalid_columns for c in feat_columns]):
            X_ = X
        else:
            columns_to_drop = [
                col for col in self.invalid_columns
                if col in X.columns
            ]
            X_ = X.drop(columns=columns_to_drop)

        num_cols = [c for c in num_cols if c in X_.columns]
        cat_cols = [c for c in cat_cols if c in X_.columns]

        self.processor = ColumnTransformer([
            ("scaler", StandardScaler(), num_cols),
            ("encoder", OrdinalEncoder(encoded_missing_value=-1), cat_cols)
        ])
        X_ = self.processor.fit_transform(X_)
        # It seems that Birch has some difficulties dealing with columns that are in different scales
        # so we standardize the the categorical values to be represented to match the numerical values
        if len(cat_cols) > 0:
            scaler = StandardScaler()
            X_[:, len(num_cols):] = scaler.fit_transform(X_[:, len(num_cols):])
        X_ = nan_to_num(X_)
        if (
            self._max_features is not None and
            len(num_cols) + len(cat_cols) > self._max_features
        ):
            X_ = PCA(self._max_features).fit_transform(X_)
        return X_

    def fit(
        self,
        X: pdDataFrame,
        metadata: Metadata,
        schema: TableSchema,
        table: str = ""
    ):
        X_ = self._preprocess(X, metadata, schema, table)
        self.model.fit(X_)

    def encode(self, X: pdDataFrame):
        X_ = X.drop(columns=[
            col for col in self.invalid_columns
            if col in X.columns
        ])
        X_ = self.processor.transform(X_)
        return self.model.predict(X_)

    def fit_predict(self, X: pdDataFrame, metadata: Metadata, schema: TableSchema, table: str = ""):
        self.fit(X, metadata, schema, table)
        return self.model.labels_

    def transform(self, X):
        return self.encode(X)


class BirchEncoder(BaseEncoder):
    def __init__(self, max_features: int = 20) -> None:
        super().__init__()
        self.model = Birch()
        self._encoded_columns = ["__cluster_id__"]
        self._max_features = max_features

    def fit(
        self,
        X: pdDataFrame,
        metadata: Metadata,
        schema: TableSchema,
        table: str = ""
    ):
        # controls the branching rate of the CFTrees
        threshold = 0.5
        data_len = len(X)
        if 10_000 < data_len <= 100_000:
            threshold = 1.5
        elif 100_000 < data_len <= 1_000_000:
            threshold = 2.5
        elif 1_000_000 < data_len:
            threshold = 3.5
        self.model = Birch(threshold=threshold, copy=False)

        super().fit(
            X=X,
            metadata=metadata,
            schema=schema,
            table=table,
        )


class AffinityPropagationEncoder(BaseEncoder):
    def __init__(self) -> None:
        super().__init__()
        self.model = AffinityPropagation()
        self._encoded_columns = ["__cluster_id__"]


class MeanShiftEncoder(BaseEncoder):
    def __init__(self) -> None:
        super().__init__()
        self.model = MeanShift()
        self._encoded_columns = ["__cluster_id__"]


class EncoderFabric:
    @staticmethod
    def create(algorithm: str | EncoderType = "birch") -> BaseEncoder:
        algorithm = EncoderType(algorithm)
        if algorithm == EncoderType.BIRCH:
            return BirchEncoder()
        if algorithm == EncoderType.AFFINITY_PROPAGATION:
            return AffinityPropagationEncoder()
        if algorithm == EncoderType.MEAN_SHIFT:
            return MeanShiftEncoder()
