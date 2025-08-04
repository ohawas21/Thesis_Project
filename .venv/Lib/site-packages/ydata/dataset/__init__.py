"""The :mod:`ydata.dataset` Model that gathers the dataset information
including its metadata."""
from ydata.dataset.dataset import Dataset
from ydata.dataset.dataset_type import DatasetType
from ydata.dataset.multidataset import MultiDataset
from ydata.dataset.textdataset import TextDataset

__all__ = ["Dataset",
           "MultiDataset",
           "DatasetType",
           "TextDataset",]
