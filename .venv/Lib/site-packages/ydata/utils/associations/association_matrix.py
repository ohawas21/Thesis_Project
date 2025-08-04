"""
The association matrix uses the MeasureAssociations class in pairwise_matrix.py
to calculate a pairwise column metric of the following mapping:
ScaleType-ScaleType : method
Nominal-Nominal     : Cramer's V
Nominal-Ordinal     : Cramer's V
Ordinal-Ordinal     : Spearman

With ScaleType.ORDINAL to currently represent both ordinal
 and ratio (i.e where the distance is significant) scale types.
"""
from __future__ import annotations

from dask.dataframe import DataFrame as ddDataFrame
from pandas import DataFrame as pdDataFrame

from ydata.dataset import Dataset
from ydata.utils.associations.mapping import mapping
from ydata.utils.associations.measure_associations import (MeasureAssociations, MeasureAssociationsPandas,
                                                           PairwiseMatrixMapping)


def association_matrix(
    dataset: Dataset | ddDataFrame | pdDataFrame,
    datatypes: dict | None = None,
    vartypes: dict | None = None,
    columns: dict | None = None,
    association_measurer: type[MeasureAssociations] = None,
    mapping: PairwiseMatrixMapping | None = mapping
) -> pdDataFrame:
    """compute the association correlation.

    Args:
        dataset (Dataset): dataset
        columns (list | None): columns to calculate the associations for. Defaults to None.
        mapping (PairwiseMatrixMapping | None): mapping. Defaults to None.

    Returns:
        pdDataFrame: association correlation matrix
    """
    if association_measurer is None:
        association_measurer = MeasureAssociationsPandas if isinstance(
            dataset, pdDataFrame) else MeasureAssociations

    association_measurer_ = association_measurer(mapping=mapping)
    association_matrix = association_measurer_.compute_pairwise_matrix(
        dataset, datatypes=datatypes, vartypes=vartypes, columns=columns)
    return association_matrix.fillna(0.)
