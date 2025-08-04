from numpy import array as array, ndarray as ndarray
from pandas import Series as pdSeries

def crosstab_by_factorization(col1, col2, sparse: bool = False): ...
def compute_chi_squared_based_values(col1: pdSeries, col2: pdSeries) -> tuple[tuple[int, int], float, int]:
    """return chi-squared based information.

    Args:
        col1 (pdSeries): first column
        col2 (pdSeries): second column

    Returns:
        Tuple[Tuple[int, int], float, int]: returns the dimensions of the contingency table,
                                            the chi-squared score, and the sample size.
    """
def compute_cramers_v(cross_t_shape: tuple[int, int], chi2: float, sample_size: int) -> float:
    """Cramer’s V is a symmetrical measure of association between two nominal
    variables.

    The score is bounded [0,1] with zero corresponding to no association
    and one corresponding to complete association.
    """
def compute_num_num_adjusted_cramers_v(cross_t_shape: tuple[int, int], chi2: float, sample_size: int) -> float:
    """Compute the biased corrected Cramer's V.

    For discretised num-num columns this bias correction was found to
    introduce much needed stability, especially around low levels of
    association
    """
def compute_adjusted_cramers_v(cross_t_shape: tuple[int, int], chi2: float, sample_size: int) -> float:
    """compute the adjusted Cramer's V for nominal variables.

    For more details read the paper: a bias-correction for Cramers V and
    Tschuprows T
    """
def corrected_values(phi2: float, r: int, c: int, sample_size: int) -> tuple[float, int, int]:
    """Return the corrected values for the adjusted Cramer's V score."""
def calculate_contingency_table(col1: pdSeries, col2: pdSeries) -> ndarray:
    """compute a contingency table.

    A contingency table is a table which details how two variables are
    interrelated by detailing the multivariate frequency distributions
    """
def calculate_chi_squared(contingency_table: ndarray) -> float:
    """compute Pearson's chi-squared test statistic."""
