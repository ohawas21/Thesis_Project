"""Ranking functions."""
import warnings
from enum import Enum

import numpy as np

from ydata.quality.labels.constants import CLIPPING_LOWER_BOUND
from ydata.quality.labels.utils import _subtract_confident_thresholds, get_normalized_entropy


def get_self_confidence_for_each_label(
    labels: np.ndarray,
    pred_probs: np.ndarray,
) -> np.ndarray:
    """Returns the self-confidence label-quality score for each datapoint.

    This is a function to compute label-quality scores for classification datasets,
    where lower scores indicate labels less likely to be correct.

    The self-confidence is the classifier's predicted probability that an example belongs to
    its given class label.

    Parameters
    ----------
    labels : np.ndarray
      And array with the dataset labels

    pred_probs : np.ndarray
      An array with the Predicted-probabilities.

    Returns
    -------
    label_scores : np.ndarray
      Contains one score (between 0 and 1) per example.
      Lower scores indicate more likely mislabeled examples.
    """
    return pred_probs[np.arange(labels.shape[0]), labels]


def get_normalized_margin_for_each_label(
    labels: np.ndarray,
    pred_probs: np.ndarray,
) -> np.ndarray:
    """Returns the "normalized margin" label-quality score for each datapoint.

    This is a function to compute label-quality scores for classification datasets,
    where lower scores indicate labels less likely to be correct.

    Normalized margin works best for finding class conditional label errors where
    there is another label in the set of classes that is clearly better than the given label.

    Parameters
    ----------
    labels : np.ndarray
        An array with the dataset labels

    pred_probs : np.ndarray
      Array with the predicted probabilities

    Returns
    -------
    label_scores : np.ndarray
      Contains one score (between 0 and 1) per example.
      Lower scores indicate more likely mislabeled examples.
    """

    self_confidence = get_self_confidence_for_each_label(labels, pred_probs)
    N, K = pred_probs.shape
    del_indices = np.arange(N) * K + labels
    max_prob_not_label = np.max(
        np.delete(pred_probs, del_indices, axis=None).reshape(N, K - 1), axis=-1
    )
    label_scores = (self_confidence - max_prob_not_label + 1) / 2
    return label_scores


def get_confidence_weighted_entropy_for_each_label(
    labels: np.ndarray, pred_probs: np.ndarray
) -> np.ndarray:
    """Returns the "confidence weighted entropy" label-quality score for each
    datapoint.

    This is a function to compute label-quality scores for classification datasets,
    where lower scores indicate labels less likely to be correct.

    "confidence weighted entropy" is defined as the normalized entropy divided by "self-confidence".
    The returned values are a transformed version of this score, in order to
    ensure values between 0-1 with lower values indicating more likely mislabeled data.

    Parameters
    ----------
    labels : np.ndarray
      Labels in the same format expected by the `~cleanlab.rank.get_label_quality_scores` function.

    pred_probs : np.ndarray
      Predicted-probabilities in the same format expected by the `~cleanlab.rank.get_label_quality_scores` function.

    Returns
    -------
    label_quality_scores : np.ndarray
      Contains one score (between 0 and 1) per example.
      Lower scores indicate more likely mislabeled examples.
    """

    self_confidence = get_self_confidence_for_each_label(labels, pred_probs)
    self_confidence = np.clip(
        self_confidence, a_min=CLIPPING_LOWER_BOUND, a_max=None)

    # Divide entropy by self confidence
    label_quality_scores = get_normalized_entropy(pred_probs) / self_confidence

    # Rescale
    clipped_scores = np.clip(label_quality_scores,
                             a_min=CLIPPING_LOWER_BOUND, a_max=None)
    label_scores = np.log(label_quality_scores + 1) / clipped_scores

    return label_scores


class RankedBy(Enum):
    """Enum class for the parameter to order indices ordered by value Default
    assumes the original dataset order."""
    DEFAULT = 'default', None
    SELF_CONFIDENCE = 'self_confidence', get_self_confidence_for_each_label
    NORMALIZED_MARGIN = 'normalized_margin', get_normalized_margin_for_each_label
    CONFIDENCE_WEIGHTED_ENTROPY = 'confidence_weighted_entropy', get_confidence_weighted_entropy_for_each_label

    @classmethod
    def from_value(cls, value):
        for member in cls:
            if member.value[0] == value:
                return member
        raise ValueError(f"{value} is not a valid value for {cls.__name__}")

    @property
    def function(self):
        return self.value[1]


def _compute_label_quality_scores(
    labels: np.ndarray,
    pred_probs: np.ndarray,
    *,
    method: RankedBy = RankedBy.SELF_CONFIDENCE,
    adjust_pred_probs: bool = False,
    confident_thresholds: np.ndarray | None = None,
) -> np.ndarray:
    """Internal implementation of get_label_quality_scores that assumes inputs
    have already been checked and are valid.

    This speeds things up. Can also take in pre-computed
    confident_thresholds to further accelerate things.
    """
    print('function')

    scoring_func = method.function

    if adjust_pred_probs:
        if method == RankedBy.CONFIDENCE_WEIGHTED_ENTROPY:
            warnings.warn(
                f"adjust_pred_probs is not currently supported for {method}. The input will be ignored")
            adjust_pred_probs = False

        pred_probs = _subtract_confident_thresholds(
            labels=labels, pred_probs=pred_probs, confident_thresholds=confident_thresholds
        )

    scoring_inputs = {"labels": labels, "pred_probs": pred_probs}
    label_quality_scores = scoring_func(**scoring_inputs)
    return label_quality_scores


def get_label_quality_scores(
    labels: np.ndarray,
    pred_probs: np.ndarray,
    *,
    method: RankedBy = RankedBy.SELF_CONFIDENCE,
    adjust_pred_probs: bool = False,
) -> np.ndarray:
    """Returns a label quality score for each datapoint. This is a function to
    compute label quality scores for standard (multi-class) classification
    datasets, where lower scores indicate labels less likely to be correct.
    Score is between 0 and 1.

    1 - clean label (given label is likely correct).
    0 - dirty label (given label is likely incorrect).

    Parameters
    ----------
    labels : np.ndarray
      A discrete vector of noisy labels, i.e. some labels may be erroneous.

    pred_probs : np.ndarray, optional
        An array with the model predicted probabilities

    method : RankedBy {"default, ""self_confidence", "normalized_margin", "confidence_weighted_entropy"}, default=RankedBy.SELF_CONFIDENCE
      Label quality scoring method.

    adjust_pred_probs : bool
      Account for class imbalance in the label-quality scoring by adjusting predicted probabilities
      via subtraction of class confident thresholds and renormalization.
      Set this to ``True`` if you prefer to account for class-imbalance.
      More details in `Northcutt et al., 2021 <https://jair.org/index.php/jair/article/view/12125>`_.

    Returns
    -------
    label_quality_scores : np.ndarray
      Contains one score (between 0 and 1) per example.
      Lower scores indicate more likely mislabeled examples.
    """
    return _compute_label_quality_scores(
        labels=labels, pred_probs=pred_probs, method=method, adjust_pred_probs=adjust_pred_probs
    )


def order_label_issues(
    label_issues_mask: np.ndarray,
    labels: np.ndarray,
    pred_probs: np.ndarray,
    *,
    rank_by: RankedBy = RankedBy.SELF_CONFIDENCE,
    adjust_pred_probs: bool = False
) -> np.ndarray:
    """Sorts label issues by label quality score.

    Default label quality score is "self_confidence".

    Parameters
    ----------
    label_issues_mask : np.ndarray
      A boolean mask for the entire dataset where ``True`` represents a label
      issue and ``False`` represents an example that is accurately labeled with
      high confidence.

    labels : np.ndarray
      An arrya or list with the dataset labels

    pred_probs : np.ndarray (shape (rows, num_classes))
      An array with the predicted probabilities for each given class.

    rank_by : RankedBy, default is set to RankedBy.SELF_CONFIDENCE
      Score by which to order label error indices (in increasing order).

    adjust_pred_probs : bool
      Account for class imbalance in the label-quality scoring by adjusting predicted probabilities
      via subtraction of class confident thresholds and renormalization.
      Set this to ``True`` if you prefer to account for class-imbalance.
      More details in `Northcutt et al., 2021 <https://jair.org/index.php/jair/article/view/12125>`_.

    Returns
    -------
    label_issues_idx : np.ndarray
      Return an array of the indices of the examples with label issues,
      ordered by the label-quality scoring method passed to `rank_by`.
    """
    # Convert bool mask to index mask
    label_issues_idx = np.arange(len(labels))[label_issues_mask]

    # Calculate label quality scores
    label_quality_scores = get_label_quality_scores(
        labels, pred_probs, method=rank_by)

    # Get label quality scores for label issues
    label_quality_scores_issues = label_quality_scores[label_issues_mask]

    return label_issues_idx[np.argsort(label_quality_scores_issues)]
