"""Utils exceptions module."""
from __future__ import absolute_import, division, print_function

import sys

class FabricExceptionMixin:
    def __init__(self, message):
        super().__init__(message)
        self.message = message
        sys.tracebacklimit = 0

    __str__ = __repr__ = lambda self: self.message

class DatasetException(FabricExceptionMixin, Exception):
    """Exception to be raised when an Exception is raised while using the Dataset and MultiDataset objects"""
    pass

class DatasetAssertionError(FabricExceptionMixin, AssertionError):
    """Assertion Error to be raised when a requested input for Dataset and MultiDataset objects fail"""
    pass


class InvalidDatasetSample(DatasetException):
    """Exception to be raised when and invalid DatasetType is provided."""
    pass

class InvalidDatasetTypeError(FabricExceptionMixin, ValueError):
    """Exception to be raised when and invalid DatasetType is provided."""
    pass

class InvalidPublishedDataset(FabricExceptionMixin, ValueError):
    """Exception to be raised whenever a user is trying to get a non-published
    dataset."""
    pass

class InvalidDatasetSchema(FabricExceptionMixin, ValueError):
    """Exception to be raised when and invalid schema is provided to a Dataset or MultiDataset object."""
    pass

class ColumnNotFoundError(FabricExceptionMixin, KeyError):
    """Exception to be raised whenever we try to access an invalid or non-
    existing column."""
    pass


class VariableTypeConversionError(FabricExceptionMixin, TypeError):
    """Exception to be raised whenever we try to convert a variable type to an
    invalid variable type."""
    pass


class VariableTypeRequestError(FabricExceptionMixin, TypeError):
    """Exception to be raised whenever the request to convert multiple variable
    types is not correct."""
    pass


class DataTypeRequestError(FabricExceptionMixin, TypeError):
    """Exception to be raised whenever the request for datatype is
    incorrect."""


class InvalidEntityColumnError(FabricExceptionMixin, TypeError):
    """Exception to be raised whenever the entity column is invalid."""


class NotEnoughRows(FabricExceptionMixin, Exception):
    """Exception raised when the training dataset has too few rows."""
    pass


class LessRowsThanColumns(FabricExceptionMixin, Warning):
    """Warning to be raised when the training dataset has less rows than
    columns."""
    pass


class SmallTrainingDataset(FabricExceptionMixin, Warning):
    """Warning to be raised when the training dataset is small."""
    pass


class IgnoredParameter(FabricExceptionMixin, Warning):
    """Warning to be raised when a parameter is not used."""
    pass

class InvalidMetadataInputError(FabricExceptionMixin, ValueError):
    """Exception to be raised when a metadata input is invalid."""
    pass
