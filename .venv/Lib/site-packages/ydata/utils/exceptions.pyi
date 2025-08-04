from _typeshed import Incomplete

class FabricExceptionMixin:
    message: Incomplete
    def __init__(self, message) -> None: ...

class DatasetException(FabricExceptionMixin, Exception):
    """Exception to be raised when an Exception is raised while using the Dataset and MultiDataset objects"""
class DatasetAssertionError(FabricExceptionMixin, AssertionError):
    """Assertion Error to be raised when a requested input for Dataset and MultiDataset objects fail"""
class InvalidDatasetSample(DatasetException):
    """Exception to be raised when and invalid DatasetType is provided."""
class InvalidDatasetTypeError(FabricExceptionMixin, ValueError):
    """Exception to be raised when and invalid DatasetType is provided."""
class InvalidPublishedDataset(FabricExceptionMixin, ValueError):
    """Exception to be raised whenever a user is trying to get a non-published
    dataset."""
class InvalidDatasetSchema(FabricExceptionMixin, ValueError):
    """Exception to be raised when and invalid schema is provided to a Dataset or MultiDataset object."""
class ColumnNotFoundError(FabricExceptionMixin, KeyError):
    """Exception to be raised whenever we try to access an invalid or non-
    existing column."""
class VariableTypeConversionError(FabricExceptionMixin, TypeError):
    """Exception to be raised whenever we try to convert a variable type to an
    invalid variable type."""
class VariableTypeRequestError(FabricExceptionMixin, TypeError):
    """Exception to be raised whenever the request to convert multiple variable
    types is not correct."""
class DataTypeRequestError(FabricExceptionMixin, TypeError):
    """Exception to be raised whenever the request for datatype is
    incorrect."""
class InvalidEntityColumnError(FabricExceptionMixin, TypeError):
    """Exception to be raised whenever the entity column is invalid."""
class NotEnoughRows(FabricExceptionMixin, Exception):
    """Exception raised when the training dataset has too few rows."""
class LessRowsThanColumns(FabricExceptionMixin, Warning):
    """Warning to be raised when the training dataset has less rows than
    columns."""
class SmallTrainingDataset(FabricExceptionMixin, Warning):
    """Warning to be raised when the training dataset is small."""
class IgnoredParameter(FabricExceptionMixin, Warning):
    """Warning to be raised when a parameter is not used."""
class InvalidMetadataInputError(FabricExceptionMixin, ValueError):
    """Exception to be raised when a metadata input is invalid."""
