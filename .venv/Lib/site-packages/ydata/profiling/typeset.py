import warnings

from pandas import Series as pdSeries
from pandas.api import types as pdt
from ydata_profiling.config import Settings
from ydata_profiling.model.typeset import ProfilingTypeSet as BaseTypeSet
from ydata_profiling.model.typeset import typeset_types


class ProfilingTypeSet(BaseTypeSet):
    def __init__(self, config: Settings, type_schema: dict = None):
        self.config = config

        types = typeset_types(config)

        # turns all numeric variables in timeseries if ts_mode is active
        if config.vars.timeseries.active:
            timeseries_type = None
            for t in types:
                if str(t) == "TimeSeries":
                    timeseries_type = t

            def is_timeseries(series: pdSeries, state: dict):
                return pdt.is_numeric_dtype(series) and not pdt.is_bool_dtype(series) and series.nunique() > 1
            timeseries_type.contains_op = is_timeseries

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            super(BaseTypeSet, self).__init__(types)

        self.type_schema = self._init_type_schema(type_schema or {})
