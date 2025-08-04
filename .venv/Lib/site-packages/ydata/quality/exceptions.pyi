from ydata.utils.exceptions import FabricExceptionMixin

class InvalidConfigurationError(FabricExceptionMixin, LookupError):
    """Exception to be raised when an invalid configuration is found."""
