
from ydata.dataset.dataset import Dataset
from ydata.metadata.metadata import Metadata


def is_entity(col: str, m: Metadata):
    if m.dataset_attrs:
        isentity = col in m.dataset_attrs.entities if m is not None else False
    else:
        isentity = False
    return isentity


def get_categorical_vars(dataset: Dataset, meta: Metadata):
    return [
        c
        for c in meta.categorical_vars
        if not is_entity(c, meta) and c in dataset.columns
    ]


def get_numerical_vars(dataset: Dataset, meta: Metadata):
    return [
        c
        for c in meta.numerical_vars
        if not is_entity(c, meta) and c in dataset.columns
    ]
