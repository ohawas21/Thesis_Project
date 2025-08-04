Axis = str | int

class Constraint:
    """Base constraint class.

    For now it is used only for typechecking.
    """
class RowConstraint(Constraint):
    """Base class for row constraint.

    A row constraint is a constraint that returns a mask per row to
    indicate a violation
    """
class ColumnConstraint(Constraint):
    """Base class for column constraint.

    A column constraint is a constraint that returns a mask per column
    to indicate a violation
    """
