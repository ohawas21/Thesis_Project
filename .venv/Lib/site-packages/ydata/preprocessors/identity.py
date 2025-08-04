from copy import deepcopy

from ydata.pipeline.identity import Identity as IdentityPipeline
from ydata.preprocessors.base import Preprocessor


class Identity(Preprocessor, IdentityPipeline):
    def __init__(self, **_):
        Preprocessor.__init__(self, steps=[("identity", None)])
        IdentityPipeline.__init__(self)

    def output_dtypes(self, step: int):
        return self._output_dtypes

    def input_dtypes(self, step: int):
        return self._input_dtypes

    def fit(self, X, input_dtypes):
        self._input_dtypes = deepcopy(input_dtypes)
        self._output_dtypes = deepcopy(input_dtypes)
        return self

    def fit_transform(self, X, input_dtypes):
        self.fit(X, input_dtypes)
        return X

    def transform(self, X):
        return X

    def inverse_transform(self, X):
        return X
