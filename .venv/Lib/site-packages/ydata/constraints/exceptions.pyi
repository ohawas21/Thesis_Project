from ydata.utils.exceptions import FabricExceptionMixin

class ConstraintError(FabricExceptionMixin, Exception):
    """Base class for all constraint exception."""
class NotValidatedError(ConstraintError):
    """Raised when a constraint or constraint engine was not validated."""
class ConstraintDoesNotExist(ConstraintError):
    """Raised when accessing a constraint which does not exist."""
