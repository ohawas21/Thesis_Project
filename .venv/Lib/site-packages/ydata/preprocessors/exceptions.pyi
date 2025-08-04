from ydata.utils.exceptions import FabricExceptionMixin

class MaxIteration(FabricExceptionMixin, StopIteration):
    """Exception raised when a maximum number of iterations has been
    reached."""
class AnonymizerMaxIteration(FabricExceptionMixin, BaseException):
    """Exception raised by the synthesizer when a maximum number of iterations
    has been reached."""
class InvalidAnonymizer(FabricExceptionMixin, BaseException):
    """Exception raised when an Invalid Anonymizer is used."""
class InvalidAnonymizerConfig(FabricExceptionMixin, BaseException):
    """Exception raised when the configuration of a Anonymizer is invalid."""
class InvalidAnonymizerInputType(FabricExceptionMixin, BaseException):
    """Exception raised when the input type of the anonymizer is non
    categorical."""
