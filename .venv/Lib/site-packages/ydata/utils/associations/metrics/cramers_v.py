"""Module to compute Cramer's V and Chi-Squared based statistical tests.

Cramer's V was devised by Harald Cramer in the book
Mathematical Methods of Statistics (1946).

The adjusted Cramer's V and Tschuprow's T has been detailed in the paper:
http://stats.lse.ac.uk/bergsma/pdf/cramerV3.pdf.
"""
from __future__ import annotations

import warnings
from typing import Tuple

from numpy import bincount, ndarray, array
from numpy import sqrt as np_sqrt
from numpy import sum as np_sum
from pandas import DataFrame as pdDataFrame
from pandas import Series as pdSeries
from pandas import factorize, MultiIndex
from scipy.sparse import csr_matrix
from scipy.stats import chi2_contingency


def crosstab_by_factorization(col1, col2, sparse=False):
    # Factorize col1 and col2
    i, r = factorize(col1)
    j, c = factorize(col2)

    # Convert (i, j) pairs to a NumPy array to avoid FutureWarning
    pairs = MultiIndex.from_arrays([i, j]).to_flat_index()
    ij, tups = factorize(pairs)

    if sparse:
        a = csr_matrix((bincount(ij), tuple(zip(*tups))))
        return pdDataFrame.sparse.from_spmatrix(a, r, c).sort_index().sort_index(axis=1)
    else:
        result = dict(zip(tups, bincount(ij)))
        return pdSeries(result).unstack(fill_value=0)


def compute_chi_squared_based_values(
    col1: pdSeries, col2: pdSeries
) -> Tuple[Tuple[int, int], float, int]:
    """return chi-squared based information.

    Args:
        col1 (pdSeries): first column
        col2 (pdSeries): second column

    Returns:
        Tuple[Tuple[int, int], float, int]: returns the dimensions of the contingency table,
                                            the chi-squared score, and the sample size.
    """
    cross_t = calculate_contingency_table(col1, col2)
    if cross_t.shape[0] < 1:
        return None, None, None
    chi2 = calculate_chi_squared(cross_t)
    sample_size = np_sum(cross_t)
    cross_t_shape = cross_t.shape
    return cross_t_shape, chi2, sample_size


def compute_cramers_v(
    cross_t_shape: Tuple[int, int], chi2: float, sample_size: int
) -> float:
    """Cramer’s V is a symmetrical measure of association between two nominal
    variables.

    The score is bounded [0,1] with zero corresponding to no association
    and one corresponding to complete association.
    """
    minimum_dimension = min(cross_t_shape) - 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # This might raise a warning
        return np_sqrt((chi2 / sample_size) / minimum_dimension)


def compute_num_num_adjusted_cramers_v(
    cross_t_shape: Tuple[int, int], chi2: float, sample_size: int
) -> float:
    """Compute the biased corrected Cramer's V.

    For discretised num-num columns this bias correction was found to
    introduce much needed stability, especially around low levels of
    association
    """
    r, c = cross_t_shape
    phi2corr, rcorr, ccorr = corrected_values(
        chi2 / sample_size, r, c, sample_size)
    if min(ccorr - 1, rcorr - 1) == 0:
        return 0
    return np_sqrt(phi2corr / min(ccorr - 1, rcorr - 1))


def compute_adjusted_cramers_v(
    cross_t_shape: Tuple[int, int], chi2: float, sample_size: int
) -> float:
    """compute the adjusted Cramer's V for nominal variables.

    For more details read the paper: a bias-correction for Cramers V and
    Tschuprows T
    """
    phi2 = chi2 / sample_size
    # where r is the number of rows, and c is the number of columns
    r, c = cross_t_shape
    phi2corr, rcorr, ccorr = corrected_values(phi2, r, c, sample_size)
    if min(ccorr - 1, rcorr - 1) == 0:
        return compute_cramers_v(cross_t_shape, chi2, sample_size)
    return np_sqrt(phi2corr / min(ccorr - 1, rcorr - 1))


def corrected_values(
    phi2: float, r: int, c: int, sample_size: int
) -> Tuple[float, int, int]:
    """Return the corrected values for the adjusted Cramer's V score."""
    phi2corr = max(0, phi2 - ((c - 1) * (r - 1)) / (sample_size - 1))
    rcorr = r - (r - 1**2) / (sample_size - 1)
    ccorr = c - (c - 1**2) / (sample_size - 1)
    return phi2corr, rcorr, ccorr


def calculate_contingency_table(col1: pdSeries, col2: pdSeries) -> ndarray:
    """compute a contingency table.

    A contingency table is a table which details how two variables are
    interrelated by detailing the multivariate frequency distributions
    """
    return crosstab_by_factorization(col1, col2, sparse=False).to_numpy()


def calculate_chi_squared(contingency_table: ndarray) -> float:
    """compute Pearson's chi-squared test statistic."""
    return chi2_contingency(contingency_table, correction=False)[0]
