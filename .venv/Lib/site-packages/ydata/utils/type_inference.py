"""Utilities to parse data from strings.

Heavily inspired by https://github.com/bruth/strconv
"""
import re
from datetime import date, datetime, time
from decimal import Decimal

import numpy as np
import pandas as pd
from pandas.core.arrays.floating import Float32Dtype, Float64Dtype
from pandas.core.arrays.integer import (Int8Dtype, Int16Dtype, Int32Dtype, Int64Dtype,
                                        UInt8Dtype, UInt16Dtype, UInt32Dtype, UInt64Dtype )

FLOAT_STR_TO_DTYPE = {
    "float32": Float32Dtype(),
    "float64": Float64Dtype(),
}

INT_STR_TO_DTYPE = {
    "int8": Int8Dtype(),
    "int16": Int16Dtype(),
    "int32": Int32Dtype(),
    "int64": Int64Dtype(),
    "uint8": UInt8Dtype(),
    "uint16": UInt16Dtype(),
    "uint32": UInt32Dtype(),
    "uint64": UInt64Dtype(),
}


class TypeInference:
    "Convert and infer types."

    def __init__(self, converters=()):
        self.converters = converters

    @property
    def inferable_types(self):
        "Returns a list of types infered for a given TypeInference instance."
        return [k[0] for k in self.converters]

    def register_converter(self, name, func):
        "Registers a converter method to the TypeInference."
        if not callable(func):
            raise ValueError("Converter function must be callable.")
        self.converters[name] = func

    def get_converter(self, name):
        "Returns a converter method by the name."
        return self.converters[name]

    def convert(self, s, include_type=False):
        "Converts a given string per the specified order in the .converters property."
        # if isinstance(s, str):
        for (name, func) in self.converters:
            try:
                v = func(str(s))
                if include_type:
                    return v, name
                return v
            except ValueError:
                pass
        if include_type:
            return s, None
        return s

    def infer(self, s):
        "Infers the data type of a given string."
        _, t = self.convert(s, include_type=True)
        return t


# ############# Standard Date and Time Formats ###########
DATE_FORMATS = (
    "%Y-%m-%d",
    "%m-%d-%Y",
    "%Y/%m/%d",
    "%m/%d/%Y",
    "%m.%d.%Y",
    "%m-%d-%y",
    "%B %d, %Y",
    "%B %d, %y",
    "%b %d, %Y",
    "%b %d, %y",
)

TIME_FORMATS = (
    "%H:%M:%S",
    "%H:%M",
    "%H:%M:%S%z",
    "%I:%M:%S %p",
    "%I:%M:%S%z",
    "%I:%M:%S %z",
    "%I:%M %p",
    "%I:%M %z",
    "%I:%M",
)

DATE_TIME_SEPS = (" ", "T")
# ############# End of Date and Time Formats ###########

# ############# Standard Boolean Formats ###########
true_re = re.compile(r"^(t(rue)?|y(es)?)$", re.I)
false_re = re.compile(r"^(f(alse)?|n(o)?)$", re.I)
# ############# End of Boolean Formats ###########


# ############# Converter Methods ###########
def convert_bool_int(s):
    if int(s) == 1:
        return True
    if int(s) == 0:
        return False
    raise ValueError


def convert_int(s):
    return int(s)


def convert_float(s):
    return float(s)


def convert_bool(s):
    if true_re.match(s):
        return True
    if false_re.match(s):
        return False
    raise ValueError


def convert_string(s):
    return str(s)


def convert_datetime(s, date_formats=DATE_FORMATS, time_formats=TIME_FORMATS):
    for df in date_formats:
        for tf in time_formats:
            for sep in DATE_TIME_SEPS:
                f = "{0}{1}{2}".format(df, sep, tf)
                try:
                    return datetime.strptime(s, f)
                except ValueError:
                    pass
    raise ValueError


def convert_date(s, date_formats=DATE_FORMATS):
    for f in date_formats:
        try:
            return datetime.strptime(s, f).date()
        except ValueError:
            pass
    raise ValueError


def convert_time(s, time_formats=TIME_FORMATS):
    for f in time_formats:
        try:
            return datetime.strptime(s, f).time()
        except ValueError:
            pass
    raise ValueError


# ############# End of Converter Methods ###########

# ############# Define a Default String Converter ###########
default_inference = TypeInference(
    converters=[
        # ('bool', convert_bool_int),
        ("int", convert_int),
        ("float", convert_float),
        ("bool", convert_bool),
        ("time", convert_time),
        ("datetime", convert_datetime),
        ("date", convert_date),
        ("string", convert_string),
    ]
)

DEFAULT_TYPES = default_inference.inferable_types
# ############# End of Default String Converter ###########


class TypeConverter:
    "Helper class to convert low-level data types implementations (numpy, pandas dtypes) to YData's intermediate dtypes."

    @staticmethod
    def to_low(dtype: str):
        "Converts intermediate to low-level. By default, cast to str."

        MAPPER = {
            "int": int,
            "float": float,
            "bool": bool,
            "time": np.dtype("datetime64"),
            "datetime": np.dtype("datetime64[ns]"),
            "date": np.dtype("datetime64"),
            "string": pd.StringDtype,
        }
        try:
            return MAPPER[dtype]
        except BaseException:
            return str

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
        #This is required as pandas as
        if isinstance(dtype, pd.StringDtype):
           dtype = type(dtype)

        MAPPER = {
            np.dtype("int64"): "int",
            np.dtype("int32"): "int",
            np.dtype("short"): "int",
            np.dtype("ushort"): "int",
            np.dtype("intc"): "int",
            np.dtype("uintc"): "int",
            np.dtype("int_"): "int",
            np.dtype("uint"): "int",
            np.dtype("longlong"): "int",
            np.dtype("ulonglong"): "int",
            np.dtype("uint8"): "int",
            np.dtype("half"): "float",
            np.dtype("float16"): "float",
            np.dtype("single"): "float",
            np.dtype("double"): "float",
            np.dtype("longdouble"): "float",
            np.dtype("float32"): "float",
            np.dtype("float64"): "float",
            np.dtype("O"): "string",
            np.dtype("bool"): "bool",
            np.dtype("bool_"): "bool",
            np.dtype("datetime64"): "datetime",
            np.dtype("timedelta64[ns]"): "int",
            np.dtype("<M8[ns]"): "datetime",
            np.dtype("<m8[ns]"): "int",  # Timedelta
            datetime: 'datetime',
            date: 'date',
            time: 'datetime',
            Decimal: 'float',
            bytes: 'string',
            int: "int",
            float: "float",
            str: "string",
            bool: "bool",
            pd.StringDtype: "string",
            'string[pyarrow]': "string",

        }
        # pandas types
        pdint = {t: "int" for t in INT_STR_TO_DTYPE.values()}
        pdfloat = {t: "float" for t in FLOAT_STR_TO_DTYPE.values()}
        MAPPER |= pdint | pdfloat

        try:
            return MAPPER[dtype]
        except KeyError:
            return dtype


def infer_MV_code(df, col):  # noqa: N802
    """Infer Missing Value code for mixed typed column in a pd.DataFrame.

    It is assumed that the last type represents the missing value code.
    Then, it is assumed that there is only one value for this particular
    data type. If this is the case, this value is considered as the
    missing value, otherwise, None is returned.
    """
    type_col = df[col].apply(type)
    type_ = type_col.value_counts().reset_index().iloc[-1][col]
    values = df[col][type_col == type_].unique()
    return values[0] if len(values) == 1 else None
