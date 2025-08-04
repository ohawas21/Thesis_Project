"""All auziliary functions that refer to the pruning method."""
import warnings
from typing import Dict

import numpy as np

from ydata.quality.labels.constants import FLOATING_POINT_COMPARISON

pred_probs_by_class: Dict[int, np.ndarray]
prune_count_matrix_cols: Dict[int, np.ndarray]


def _reduce_prune_counts(prune_count_matrix: np.ndarray, frac_noise: float = 1.0) -> np.ndarray:
    """Reduce (multiply) all prune counts (non-diagonal) by frac_noise and
    increase diagonal by the total amount reduced in each column to preserve
    column counts.

    Parameters
    ----------
    prune_count_matrix : np.ndarray of shape (K, K), K = number of classes
        A counts of mislabeled examples in every class. For this function, it
        does not matter what the rows or columns are, but the diagonal terms
        reflect the number of correctly labeled examples.

    frac_noise : float
      Used to only return the "top" ``frac_noise * num_label_issues``. The choice of which "top"
      label issues to return is dependent on the `filter_by` method used. It works by reducing the
      size of the off-diagonals of the `prune_count_matrix` of given labels and true labels
      proportionally by `frac_noise` prior to estimating label issues with each method.
      When frac_noise=1.0, return all "confident" estimated noise indices (recommended).
    """

    new_mat = prune_count_matrix * frac_noise
    np.fill_diagonal(new_mat, prune_count_matrix.diagonal())
    np.fill_diagonal(
        new_mat,
        prune_count_matrix.diagonal() + np.sum(prune_count_matrix - new_mat, axis=0),
    )
    # These are counts, so return a matrix of ints.
    return new_mat.astype(int)


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

    floats = np.asarray(iterable, dtype=float)
    ints = floats.round()
    orig_sum = np.sum(floats).round()
    int_sum = np.sum(ints).round()
    # Adjust the integers so that they sum to orig_sum
    while abs(int_sum - orig_sum) > FLOATING_POINT_COMPARISON:
        diff = np.round(orig_sum - int_sum)
        increment = -1 if int(diff < 0.0) else 1
        changes = min(int(abs(diff)), len(iterable))
        # Orders indices by difference. Increments # of changes.
        indices = np.argsort(floats - ints)[::-increment][:changes]
        for i in indices:
            ints[i] = ints[i] + increment
        int_sum = np.sum(ints).round()
    return ints.astype(int)


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

    return np.apply_along_axis(
        func1d=round_preserving_sum,
        axis=1,
        arr=confident_joint,
    ).astype(int)


def _keep_at_least_n_per_class(
    prune_count_matrix: np.ndarray, n: int, *, frac_noise: float = 1.0
) -> np.ndarray:
    """Make sure every class has at least n examples after removing noise.
    Functionally, increase each column, increases the diagonal term #(true_label=k,label=k)
    of prune_count_matrix until it is at least n, distributing the amount
    increased by subtracting uniformly from the rest of the terms in the
    column. When frac_noise = 1.0, return all "confidently" estimated
    noise indices, otherwise this returns frac_noise fraction of all
    the noise counts, with diagonal terms adjusted to ensure column
    totals are preserved.

    Parameters
    ----------
    prune_count_matrix : np.ndarray of shape (K, K), K = number of classes
        A counts of mislabeled examples in every class. For this function.
        NOTE prune_count_matrix is transposed relative to confident_joint.

    n : int
        Number of examples to make sure are left in each class.

    frac_noise : float, default=1.0
      Used to only return the "top" ``frac_noise * num_label_issues``. The choice of which "top"
      label issues to return is dependent on the `filter_by` method used. It works by reducing the
      size of the off-diagonals of the `prune_count_matrix` of given labels and true labels
      proportionally by `frac_noise` prior to estimating label issues with each method.
      When frac_noise=1.0, return all "confident" estimated noise indices (recommended).

    Returns
    -------
    prune_count_matrix : np.ndarray of shape (K, K), K = number of classes
        This the same as the confident_joint, but has been transposed and the counts are adjusted.
    """

    prune_count_matrix_diagonal = np.diagonal(prune_count_matrix)

    # Set diagonal terms less than n, to n.
    new_diagonal = np.maximum(prune_count_matrix_diagonal, n)

    # Find how much diagonal terms were increased.
    diff_per_col = new_diagonal - prune_count_matrix_diagonal

    # Count non-zero, non-diagonal items per column
    # np.maximum(*, 1) makes this never 0 (we divide by this next)
    num_noise_rates_per_col = np.maximum(
        np.count_nonzero(prune_count_matrix, axis=0) - 1.0,
        1.0,
    )

    # Uniformly decrease non-zero noise rates by the same amount
    # that the diagonal items were increased
    new_mat = prune_count_matrix - diff_per_col / num_noise_rates_per_col

    # Originally zero noise rates will now be negative, fix them back to zero
    new_mat[new_mat < 0] = 0

    # Round diagonal terms (correctly labeled examples)
    np.fill_diagonal(new_mat, new_diagonal)

    # Reduce (multiply) all noise rates (non-diagonal) by frac_noise and
    # increase diagonal by the total amount reduced in each column
    # to preserve column counts.
    new_mat = _reduce_prune_counts(new_mat, frac_noise)

    # These are counts, so return a matrix of ints.
    return round_preserving_row_totals(new_mat).astype(int)


def _prune_by_class(args: list) -> np.ndarray:
    """multiprocessing Helper function for find_label_issues() that assumes
    globals and produces a mask for class k for each example by removing the
    examples with *smallest probability* of belonging to their given class
    label.

    Parameters
    ----------
    k : int (between 0 and num classes - 1)
      The class of interest.
    """

    k, min_examples_per_class, arrays = args
    if arrays is None:
        pred_probs = pred_probs_by_class[k]  # noqa: F821
        prune_count_matrix = prune_count_matrix_cols[k]  # noqa: F821
    else:
        pred_probs = arrays[0]
        prune_count_matrix = arrays[1]

    label_counts = pred_probs.shape[0]
    label_issues = np.zeros(label_counts, dtype=bool)
    if label_counts > min_examples_per_class:  # No prune if not at least min_examples_per_class
        num_issues = label_counts - prune_count_matrix[k]
        # Get return_indices_ranked_by of the smallest prob of class k for examples with noisy label k
        # rank = np.partition(class_probs, num_issues)[num_issues]
        if num_issues >= 1:
            class_probs = pred_probs[:, k]
            order = np.argsort(class_probs)
            label_issues[order[:num_issues]] = True
        return label_issues

    warnings.warn(
        f"May not flag all label issues in class: {k}, it has too few examples (see argument: `min_examples_per_class`)"
    )
    return label_issues
