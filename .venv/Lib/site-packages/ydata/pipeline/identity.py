from ydata.pipeline.base import Pipeline


class Identity(Pipeline):
    def __init__(self):
        super().__init__(steps=[("identity", None)])
        self._input_dtypes = None
        self._output_dtypes = None

    def fit(self, X, input_dtypes):
        self._input_dtypes = input_dtypes
        self._output_dtypes = input_dtypes
        return self

    def fit_transform(self, X, input_dtypes):
        self.fit(X, input_dtypes)
        return X

    def transform(self, X):
        return X

    def inverse_transform(self, X):
        return X
