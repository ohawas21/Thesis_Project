from _typeshed import Incomplete

FLOAT_STR_TO_DTYPE: Incomplete
INT_STR_TO_DTYPE: Incomplete

class TypeInference:
    """Convert and infer types."""
    converters: Incomplete
    def __init__(self, converters=()) -> None: ...
    @property
    def inferable_types(self):
        """Returns a list of types infered for a given TypeInference instance."""
    def register_converter(self, name, func) -> None:
        """Registers a converter method to the TypeInference."""
    def get_converter(self, name):
        """Returns a converter method by the name."""
    def convert(self, s, include_type: bool = False):
        """Converts a given string per the specified order in the .converters property."""
    def infer(self, s):
        """Infers the data type of a given string."""

DATE_FORMATS: Incomplete
TIME_FORMATS: Incomplete
DATE_TIME_SEPS: Incomplete
true_re: Incomplete
false_re: Incomplete

def convert_bool_int(s): ...
def convert_int(s): ...
def convert_float(s): ...
def convert_bool(s): ...
def convert_string(s): ...
def convert_datetime(s, date_formats=..., time_formats=...): ...
def convert_date(s, date_formats=...): ...
def convert_time(s, time_formats=...): ...

default_inference: Incomplete
DEFAULT_TYPES: Incomplete

class TypeConverter:
    """Helper class to convert low-level data types implementations (numpy, pandas dtypes) to YData's intermediate dtypes."""
    @staticmethod
    def to_low(dtype: str):
        """Converts intermediate to low-level. By default, cast to str."""
    @staticmethod
    def from_low(dtype):
        """Converts low-level to intermediate. By default, return original.

        Example:
            >>> import numpy as np
            >>> sanitize_dtypes(np.dtype('O'))
            'string'
            >>> sanitize_dtypes(int)
            'int'
            >>> sanitize_dtypes(np.dtype('float64'))
            'float'
            >>> sanitize_dtypes(np.dtype('int64'))
            'int'
        """

def infer_MV_code(df, col):
    """Infer Missing Value code for mixed typed column in a pd.DataFrame.

    It is assumed that the last type represents the missing value code.
    Then, it is assumed that there is only one value for this particular
    data type. If this is the case, this value is considered as the
    missing value, otherwise, None is returned.
    """
