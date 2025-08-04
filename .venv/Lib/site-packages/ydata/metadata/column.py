from typing import Tuple

from ydata.characteristics import ColumnCharacteristic
from ydata.utils.data_types import DATA_VARTYPE_MAP, DataType, VariableType


class Column:
    def __init__(
        self,
        name: str,
        datatype: DataType,
        vartype: VariableType,
        characteristics: Tuple[ColumnCharacteristic, ...] = None
    ):
        self.name = name
        self.datatype = datatype
        self.vartype = vartype
        self._characteristics = characteristics if characteristics is not None else tuple()

    @property
    def datatype(self):
        return self._datatype

    @property
    def vartype(self):
        return self._vartype

    @property
    def characteristics(self) -> Tuple[ColumnCharacteristic, ...]:
        return self._characteristics

    @characteristics.setter
    def characteristics(self, characteristics: Tuple[ColumnCharacteristic, ...]):
        self._characteristics = characteristics

    @datatype.setter
    def datatype(self, value: DataType):
        self._datatype = value

    @vartype.setter
    def vartype(self, vartype: VariableType) -> VariableType:
        _error_message = (
            "Variable type " + vartype.value + ' is not valid for the data type "{}". '
            "Please select one of the following variable types {}"
        )
        valid_vartypes = DATA_VARTYPE_MAP[self.datatype]
        if vartype not in valid_vartypes:
            raise Exception(_error_message.format(
                self.datatype.value, valid_vartypes))
        self._vartype = vartype

    def __str__(self) -> str:
        str_char = "[{}]".format(
            ','.join([c.value for c in self.characteristics]))
        return f"Column(name={self.name}, vartype={self.vartype.value}, datatype={self.datatype.value}, characteristics={str_char}]"

    def __repr__(self) -> str:
        return str(self)
