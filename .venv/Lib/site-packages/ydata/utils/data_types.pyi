from _typeshed import Incomplete
from enum import Enum
from ydata.characteristics import ColumnCharacteristic

class DataType(Enum):
    NUMERICAL = 'numerical'
    CATEGORICAL = 'categorical'
    DATE = 'date'
    STR = 'string'
    LONGTEXT = 'longtext'

class VariableType(Enum):
    INT = 'int'
    FLOAT = 'float'
    STR = 'string'
    BOOL = 'bool'
    DATETIME = 'datetime'
    DATE = 'date'

class ScaleType(Enum):
    METRIC = 'metric'
    ORDINAL = 'ordinal'
    NOMINAL = 'nominal'

DATA_VARTYPE_MAP: Incomplete
CATEGORICAL_DTYPES: Incomplete

def type_check(data, _type, extra_msg: str = '') -> None:
    """Guarantees that data is of a specified type."""
def validate_datatypes(data_type: dict, valid_dtypes: list = None):
    """Guarantees that a provided data_type dict is valid."""
def is_characteristic_type_valid(characteristic: ColumnCharacteristic, vartype: VariableType) -> bool:
    """Checks if the characteristic is supported for the `vartype`.

    Args:
        characteristic (ColumnCharacteristic): a column characteristic.
        vartype (VariableType): a variable type

    Returns:
        bool: True if the characteristic is allowed for the vartype, False otherwise
    """
