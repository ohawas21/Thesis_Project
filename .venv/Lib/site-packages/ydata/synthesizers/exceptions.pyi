from ydata.utils.exceptions import FabricExceptionMixin

class SegmentSizeWarning(FabricExceptionMixin, Warning): ...
class PreprocessorException(FabricExceptionMixin, Exception):
    """Base Exception to be raised when an error happens with the data
    preprocessing step of the synthesis."""
class SynthesizerAssertionError(FabricExceptionMixin, AssertionError):
    """
    AssertionError exception to be raised when an assertion in the Synthesizers is not met
    """
class SynthesizerValueError(FabricExceptionMixin, ValueError):
    """Base Exception to be raised when a value error happens in the synthesizers"""
class SynthesizerException(FabricExceptionMixin, Exception):
    """Base Exception class to be raised when an error related with the
    synthesis algorithm happens."""
class MissingDeviceException(FabricExceptionMixin, Exception):
    """Exception to be raise when acceleration device is not set."""
class NotFittedException(SynthesizerException, ValueError, AttributeError):
    """Thrown when attempting to synthesize from a non previously fitter Synthesizer"""
class SegmentationStrategyException(FabricExceptionMixin, Exception):
    """Exception to be raised when the segmentation strategy is wrongly
    configured or cannot be determined."""
class SampleSizeException(SynthesizerException):
    """Exception to be raised if the sample size is larger than the training
    set for TimeSeriesSynthesizer."""
class TemporaryPathException(SynthesizerException):
    """Exception to be raised if we cannot create the temporary path in which
    the blocks are serialized."""
class NoInputDataTypesWarning(FabricExceptionMixin, Warning):
    """Warning to be raised when a synthesizer has no intput dtypes argument.

    In this case, metadata object is used as default.
    """
