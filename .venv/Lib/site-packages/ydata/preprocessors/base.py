from sklearn.base import BaseEstimator, TransformerMixin

from ydata.__serializer import SerializerMixin
from ydata.pipeline.base import Pipeline


class BaseOperator(BaseEstimator, TransformerMixin, SerializerMixin):
    def __init__(self):
        self._input_dtypes = None
        self._output_dtypes = None

    @property
    def input_dtypes(self):
        return self._input_dtypes

    @property
    def output_dtypes(self):
        return self._output_dtypes

    def fit(self, X, input_dtypes=None):
        raise NotImplementedError

    def fit_transform(self, X, input_dtypes=None):
        raise NotImplementedError

    def transform(self, X):
        raise NotImplementedError

    def inverse_transform(self, X):
        raise NotImplementedError


class Preprocessor(BaseOperator):
    def __init__(self, steps):
        super().__init__()
        self._steps = steps
        self._pipeline = Pipeline(steps)

    @property
    def pipeline(self):
        return self._pipeline

    def get_step(self, name: str):
        filtered_steps = [s for s in self._steps if s[0] == name]
        if len(filtered_steps) == 0:
            return None
        return filtered_steps[0][1]

    def output_dtypes(self, step: int):
        return self._pipeline[step]._output_dtypes

    def input_dtypes(self, step: int):
        return self._pipeline[step]._input_dtypes

    def fit(self, X, input_dtypes=None):
        return self._pipeline.fit(X, input_dtypes=input_dtypes)

    def transform(self, X):
        return self._pipeline.transform(X)

    def fit_transform(self, X, input_dtypes=None, **fit_params):
        return self._pipeline.fit_transform(
            X,
            input_dtypes=input_dtypes,
            **fit_params
        )

    def inverse_transform(self, X):
        return self._pipeline.inverse_transform(X)
