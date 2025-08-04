from typing import Dict, Union, final

Axis = Union[str, int]


class Constraint:
    """Base constraint class.

    For now it is used only for typechecking.
    """

    pass


class RowConstraint(Constraint):
    """Base class for row constraint.

    A row constraint is a constraint that returns a mask per row to
    indicate a violation
    """

    _AXIS_TO_AXIS_NUMBER: Dict[Axis, int] = {
        0: 0, "rows": 1, "column": 0, 1: 1}

    @final
    @classmethod
    def _get_axis_number(cls, axis: Axis) -> int:
        try:
            return cls._AXIS_TO_AXIS_NUMBER[axis]
        except KeyError:
            raise ValueError(
                f"No axis named {axis} for object type {cls.__name__}")


class ColumnConstraint(Constraint):
    """Base class for column constraint.

    A column constraint is a constraint that returns a mask per column
    to indicate a violation
    """

    pass
