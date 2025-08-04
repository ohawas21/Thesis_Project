"""Auxiliary functions."""
from typing import Union

import numpy as np
from scipy.special import xlogy

from ydata.quality.labels.constants import CONFIDENT_THRESHOLDS_LOWER_BOUND, FLOATING_POINT_COMPARISON


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


def get_confident_thresholds(
    labels: Union[list, np.ndarray],
    pred_probs: np.ndarray,
) -> np.ndarray:
    """Returns expected (average) "self-confidence" for each class. The
    confident class threshold for a class j is the expected (average) "self-
    confidence" for class j, i.e. the model-predicted probability of this class
    averaged amongst all examples labeled as class j.

    Parameters
    ----------
    labels : np.ndarray or list
      Given class labels for each example in the dataset, some of which may be erroneous,

    pred_probs : np.ndarray
      Model-predicted class probabilities for each example in the dataset,

    Returns
    -------
    confident_thresholds : np.ndarray
      An array of shape ``(K, )`` where K is the number of classes.
    """
    # When all_classes != unique_classes the class threshold for the missing classes is set to
    # BIG_VALUE such that no valid prob >= BIG_VALUE (no example will be counted in missing classes)
    # REQUIRES: pred_probs.max() >= 1

    all_classes = range(pred_probs.shape[1])  # validate this as well
    # Get the unique labels values. This can be inferred with the metadata
    unique_classes = set(labels)
    BIG_VALUE = 2

    confident_thresholds = [
        np.mean(pred_probs[:, k][labels == k]
                ) if k in unique_classes else BIG_VALUE
        for k in all_classes
    ]
    confident_thresholds = np.clip(
        confident_thresholds, a_min=CONFIDENT_THRESHOLDS_LOWER_BOUND, a_max=None
    )
    return confident_thresholds


def _subtract_confident_thresholds(
    labels: np.ndarray | None,
    pred_probs: np.ndarray,
    confident_thresholds: np.ndarray | None = None,
) -> np.ndarray:
    """Return adjusted predicted probabilities by subtracting the class
    confident thresholds and renormalizing.

    The confident class threshold for a class j is the expected (average) "self-confidence" for class j.
    The purpose of this adjustment is to handle class imbalance.

    Parameters
    ----------
    labels : np.ndarray
      Labels in the same format expected by the `cleanlab.count.get_confident_thresholds()` method.
      If labels is None, confident_thresholds needs to be passed in as it will not be calculated.
    pred_probs : np.ndarray (shape (N, K))
      Predicted-probabilities in the same format expected by the `cleanlab.count.get_confident_thresholds()` method.
    confident_thresholds : np.ndarray (shape (K,))
      Pre-calculated confident thresholds. If passed in, function will subtract these thresholds instead of calculating
      confident_thresholds from the given labels and pred_probs.

    Returns
    -------
    pred_probs_adj : np.ndarray (float)
      Adjusted pred_probs.
    """
    # Get expected (average) self-confidence for each class
    if confident_thresholds is None:
        if labels is None:
            raise ValueError(
                "Cannot calculate confident_thresholds without labels. Pass in either labels or already calculated "
                "confident_thresholds parameter. "
            )
        confident_thresholds = get_confident_thresholds(labels, pred_probs)

    # Subtract the class confident thresholds
    pred_probs_adj = pred_probs - confident_thresholds

    # Re-normalize by shifting data to take care of negative values from the subtraction
    pred_probs_adj += confident_thresholds.max()
    pred_probs_adj /= pred_probs_adj.sum(axis=1, keepdims=True)

    return pred_probs_adj


def get_normalized_entropy(
        pred_probs: np.ndarray) -> np.ndarray:
    """Return the normalized entropy of pred_probs.

    Normalized entropy is between 0 and 1. Higher values of entropy indicate higher uncertainty in the model's prediction of the correct label.
    Unlike label-quality scores, entropy only depends on the model's predictions, not the given label.

    Parameters
    ----------
    pred_probs : np.ndarray (shape (N, K))
      Each row of this matrix corresponds to an example x and contains the model-predicted
      probabilities that x belongs to each possible class: P(label=k|x)

    Returns
    -------
    entropy : np.ndarray (shape (N, ))
      Each element is the normalized entropy of the corresponding row of ``pred_probs``.

    Raises
    ------
    ValueError
        An error is raised if any of the probabilities is not in the interval [0, 1].
    """
    if np.any(pred_probs < 0) or np.any(pred_probs > 1):
        raise ValueError(
            "All probabilities are required to be in the interval [0, 1].")
    num_classes = pred_probs.shape[1]

    # Note that dividing by log(num_classes) changes the base of the log which rescales entropy to 0-1 range
    return -np.sum(xlogy(pred_probs, pred_probs), axis=1) / np.log(num_classes)


def _reduce_issues(pred_probs, labels):
    """Returns a boolean mask denoting correct predictions or predictions
    within a margin around 0.5 for binary classification, suitable for
    filtering out indices in 'is_label_issue'."""
    pred_probs_copy = np.copy(pred_probs)  # Make a copy of the original array
    pred_probs_copy[np.arange(len(labels)),
                    labels] += FLOATING_POINT_COMPARISON
    pred = pred_probs_copy.argmax(axis=1)
    mask = pred == labels
    del pred_probs_copy  # Delete copy
    return mask
