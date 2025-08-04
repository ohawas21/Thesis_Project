import numpy as np
from enum import Enum

def get_self_confidence_for_each_label(labels: np.ndarray, pred_probs: np.ndarray) -> np.ndarray:
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
def get_normalized_margin_for_each_label(labels: np.ndarray, pred_probs: np.ndarray) -> np.ndarray:
    '''Returns the "normalized margin" label-quality score for each datapoint.

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
    '''
def get_confidence_weighted_entropy_for_each_label(labels: np.ndarray, pred_probs: np.ndarray) -> np.ndarray:
    '''Returns the "confidence weighted entropy" label-quality score for each
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
    '''

class RankedBy(Enum):
    """Enum class for the parameter to order indices ordered by value Default
    assumes the original dataset order."""
    DEFAULT = ('default', None)
    SELF_CONFIDENCE = ...
    NORMALIZED_MARGIN = ...
    CONFIDENCE_WEIGHTED_ENTROPY = ...
    @classmethod
    def from_value(cls, value): ...
    @property
    def function(self): ...

def get_label_quality_scores(labels: np.ndarray, pred_probs: np.ndarray, *, method: RankedBy = ..., adjust_pred_probs: bool = False) -> np.ndarray:
    '''Returns a label quality score for each datapoint. This is a function to
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
    '''
def order_label_issues(label_issues_mask: np.ndarray, labels: np.ndarray, pred_probs: np.ndarray, *, rank_by: RankedBy = ..., adjust_pred_probs: bool = False) -> np.ndarray:
    '''Sorts label issues by label quality score.

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
    '''
