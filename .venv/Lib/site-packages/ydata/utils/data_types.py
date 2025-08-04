from enum import Enum

from ydata.characteristics import ColumnCharacteristic


class DataType(Enum):
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    DATE = "date"
    STR = "string"
    LONGTEXT = "longtext"


class VariableType(Enum):
    INT = "int"
    FLOAT = "float"
    STR = "string"
    BOOL = "bool"
    DATETIME = "datetime"
    DATE = "date"


class ScaleType(Enum):
    METRIC = "metric"
    ORDINAL = "ordinal"
    NOMINAL = "nominal"


_VALID_DATATYPES = [d for d in DataType]
_VALID_VARIABLETYPES = [v for v in VariableType]
_CATEGORICAL_VARTYPES = [
    VariableType.STR,
    VariableType.INT,
    VariableType.BOOL,
    VariableType.FLOAT,
    VariableType.DATETIME,
    VariableType.DATE
]
_NUMERICAL_VARTYPES = [VariableType.INT, VariableType.FLOAT]
_DATE_VARTYPES = [VariableType.DATE, VariableType.DATETIME]

DATA_VARTYPE_MAP = {
    DataType.CATEGORICAL: _CATEGORICAL_VARTYPES,
    DataType.NUMERICAL: _NUMERICAL_VARTYPES,
    DataType.DATE: _DATE_VARTYPES,
    DataType.STR: [VariableType.STR, VariableType.BOOL, VariableType.DATE, VariableType.DATETIME],
    DataType.LONGTEXT: [VariableType.STR],
}

CATEGORICAL_DTYPES = [DataType.CATEGORICAL, DataType.STR]

# Automatically define a map between DataTypes and VariableTypes to help with type checking
_DT_TO_VT = {dt: globals().get(f"_{dt.name}_VARTYPES", [])
             for dt in _VALID_DATATYPES}

_NULL_VALUES = {
    "#N/A",
    "#N/A N/A",
    "#NA",
    "-1.#IND",
    "-1.#QNAN",
    "-NaN",
    "-nan",
    "1.#IND",
    "1.#QNAN",
    "<NA>",
    "N/A",
    "NA",
    "NULL",
    "NaN",
    "n/a",
    "nan",
    "null",
    "",
    None,
}


_CHARACTERISTIC_VARTYPE = {
    ColumnCharacteristic.ID: {VariableType.STR, VariableType.INT, VariableType.FLOAT},
    ColumnCharacteristic.EMAIL: {VariableType.STR},
    ColumnCharacteristic.URL: {VariableType.STR},
    ColumnCharacteristic.UUID: {VariableType.STR},
    ColumnCharacteristic.NAME: {VariableType.STR},
    ColumnCharacteristic.PHONE: {VariableType.STR, VariableType.INT, VariableType.FLOAT},
    ColumnCharacteristic.VAT: {VariableType.STR, VariableType.INT, VariableType.FLOAT},
    ColumnCharacteristic.IBAN: {VariableType.STR},
    ColumnCharacteristic.CREDIT_CARD: {VariableType.STR, VariableType.INT, VariableType.FLOAT},
    ColumnCharacteristic.COUNTRY: {VariableType.STR},
    ColumnCharacteristic.ZIPCODE: {VariableType.STR, VariableType.INT, VariableType.FLOAT},
    ColumnCharacteristic.ADDRESS: {VariableType.STR},
    ColumnCharacteristic.PII: set(),
    ColumnCharacteristic.LOCATION: set(),
    ColumnCharacteristic.PERSON: set(),
}


def type_check(data, _type, extra_msg=""):
    """Guarantees that data is of a specified type."""
    # TODO: Enable Union of types
    if not isinstance(data, _type):
        raise Exception(
            f"Expected {_type.__name__}, got {type(data).__name__}." +
            " " + extra_msg
        )


def validate_datatypes(data_type: dict, valid_dtypes: list = None):
    """Guarantees that a provided data_type dict is valid."""
    if valid_dtypes is None:
        valid_dtypes = _VALID_DATATYPES

    valid_columns = []
    for k, v in data_type.items():
        if DataType(v) in valid_dtypes:
            valid_columns.append(k)
    return valid_columns


def is_characteristic_type_valid(
    characteristic: ColumnCharacteristic,
    vartype: VariableType
) -> bool:
    """Checks if the characteristic is supported for the `vartype`.

    Args:
        characteristic (ColumnCharacteristic): a column characteristic.
        vartype (VariableType): a variable type

    Returns:
        bool: True if the characteristic is allowed for the vartype, False otherwise
    """
    return vartype in _CHARACTERISTIC_VARTYPE[characteristic]
