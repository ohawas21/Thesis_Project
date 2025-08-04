"""
    Synthesizers Preprocessor exceptions
"""
from ydata.utils.exceptions import FabricExceptionMixin

class MaxIteration(FabricExceptionMixin, StopIteration):
    """Exception raised when a maximum number of iterations has been
    reached."""
    pass


class AnonymizerMaxIteration(FabricExceptionMixin, BaseException):
    """Exception raised by the synthesizer when a maximum number of iterations
    has been reached."""
    pass


class InvalidAnonymizer(FabricExceptionMixin, BaseException):
    """Exception raised when an Invalid Anonymizer is used."""
    pass

class InvalidAnonymizerConfig(FabricExceptionMixin, BaseException):
    """Exception raised when the configuration of a Anonymizer is invalid."""
    pass

class InvalidAnonymizerInputType(FabricExceptionMixin, BaseException):
    """Exception raised when the input type of the anonymizer is non
    categorical."""
    pass
