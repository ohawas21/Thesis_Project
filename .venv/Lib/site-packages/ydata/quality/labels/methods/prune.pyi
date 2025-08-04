import numpy as np

pred_probs_by_class: dict[int, np.ndarray]
prune_count_matrix_cols: dict[int, np.ndarray]

def round_preserving_sum(iterable) -> np.ndarray:
    """Rounds an iterable of floats while retaining the original summed value.
    The name of each parameter is required. The type and description of each
    parameter is optional, but should be included if not obvious.

    The while loop in this code was adapted from:
    https://github.com/cgdeboer/iteround

    Parameters
    -----------
    iterable : list<float> or np.ndarray<float>
        An iterable of floats

    Returns
    -------
    list<int> or np.ndarray<int>
        The iterable rounded to int, preserving sum.
    """
def round_preserving_row_totals(confident_joint) -> np.ndarray:
    """Rounds confident_joint cj to type int while preserving the totals of
    reach row. Assumes that cj is a 2D np.ndarray of type float.

    Parameters
    ----------
    confident_joint : 2D np.ndarray<float> of shape (K, K)
        See compute_confident_joint docstring for details.

    Returns
    -------
    confident_joint : 2D np.ndarray<int> of shape (K,K)
        Rounded to int while preserving row totals.
    """
