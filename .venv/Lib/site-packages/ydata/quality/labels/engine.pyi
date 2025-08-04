import numpy as np
import pandas as pd
from _typeshed import Incomplete
from sklearn.base import BaseEstimator, TransformerMixin
from ydata.dataset import Dataset
from ydata.metadata import Metadata as Metadata
from ydata.quality.labels.enums import LabelFilter
from ydata.quality.labels.methods.rank import RankedBy

def get_unique_classes(labels, multi_label: Incomplete | None = None) -> set:
    """Returns the set of unique classes for both single-labeled and multi-
    labeled labels.

    If multi_label is set to None (default) this method will infer if
    multi_label is True or False based on the format of labels. This
    allows for a more general form of multiclass labels that looks like
    this: [1, [1,2], [0], [0, 1], 2, 1]
    """
def num_unique_classes(labels, multi_label: Incomplete | None = None) -> int:
    """Finds the number of unique classes for both single-labeled and multi-
    labeled labels.

    If multi_label is set to None (default) this method will infer if
    multi_label is True or False based on the format of labels. This
    allows for a more general form of multiclass labels that looks like
    this: [1, [1,2], [0], [0, 1], 2, 1]
    """
def get_num_classes(labels: Incomplete | None = None, pred_probs: Incomplete | None = None, label_matrix: Incomplete | None = None, multi_label: Incomplete | None = None) -> int:
    """Determines the number of classes based on information considered in a
    canonical ordering.

    label_matrix can be: noise_matrix, inverse_noise_matrix, confident_joint,
    or any other K x K matrix where K = number of classes.
    """

class FindInconsistentLabelsEngine(BaseEstimator, TransformerMixin):
    filter_type: Incomplete
    ranked_by: Incomplete
    frac_noise: Incomplete
    num_to_remove_per_class: Incomplete
    min_examples_per_class: Incomplete
    def __init__(self, filter_type: str | LabelFilter = ..., frac_noise: float = 1.0, num_to_remove_per_class: list[int] | None = None, min_examples_per_class: int = 1, indices_ranked_by: str | RankedBy = ...) -> None:
        '''Identifies potentially bad labels in a classification dataset using
        different filtering methods like pruning and confident learning.

        This class return a boolean mask for the entire dataset where ``True`` represents
        an example identified with a label issue and ``False`` represents an example that seems correctly labeled.

        Instead of a mask, you can obtain indices of the examples with label issues in your dataset
        (sorted by issue severity) by specifying the `return_indices_ranked_by` argument.
        This determines which label quality score is used to quantify severity,
        and is useful to view only the top-`J` most severe issues in your dataset.

        :param filter_type: FilterType (\'prune_by_noise_rate\', \'confident_learning\'), default is set to FilterType.PRUNE
            - FilterType.PRUNE : filters examples with high probability of being mislabeled for every non-diagonal the
            confident joint (see `prune_counts_matrix` in `filter.py`). These are the examples where (with high confidence) the given label is unlikely to match the predicted label for the example.
            - FilterType.CONFIDENT_LEARNING: filters the examples counted as part of the off-diagonals of the confident joint.
            These are the examples that are confidently predicted to be a different label than their given label.
        :param frac_noise: float, default=1.0
            Used to only return the "top" ``frac_noise * num_label_issues``. The choice of which "top"
            label issues to return is dependent on the `filter_by` method used. It works by reducing the
            size of the off-diagonals of the `joint` distribution of given labels and true labels
            proportionally by `frac_noise` prior to estimating label issues with each method.
        :param num_to_remove_per_class: array_like
              An iterable of length n_classes. Only used when filter_type==FilterType.PRUNE.
              E.g. if n_classes= 3, ``num_to_remove_per_class=[5, 0, 1]`` would return
              the indices of the 5 most likely mislabeled examples in class 0,
              and the most likely mislabeled example in class 2.
        :param min_examples_per_class: int, default=1
              Minimum number of examples per class to avoid flagging as label issues.
              This parameter can be used to avoid that all examples from a class are removed, particularly
              when pruning examples from rare classes.
        :param indices_ranked_by:RankedBy (default, self_confidence, normalized_margin, confidence_weighted_entropy)
            This parameter determines what is expected to be returned by this class transform: either a boolean mask or a list of indices np.darray.
            If default, then a boolean mask is returned.
            If other option is selected, the class will return a sorted array of indices with the examples of the labels
            with issues. Indices are sorted by label quality score.

        :param n_jobs:
        '''
    confident_joint: Incomplete
    n_classes: Incomplete
    big_dataset: Incomplete
    label_counts: Incomplete
    def fit(self, X: Dataset | pd.DataFrame, metadata: Metadata, confident_joint: np.ndarray | None = None): ...
    def transform(self, X: Dataset | pd.DataFrame, label_name: str, pred_probs: np.ndarray, n_jobs: int | None = None):
        """
        :param dataset:
        :param labels: np.ndarray or list
          A discrete vector of noisy labels for a classification dataset, i.e. some labels may be erroneous.
          *Format requirements*: for dataset with K classes, each label must be integer in 0, 1, ..., K-1.
          For a standard (multi-class) classification dataset where each example is labeled with one class,
          `labels` should be 1D array of shape ``(N,)``, for example: ``labels = [1,0,2,1,1,0...]``.
        :param pred_probs: np.ndarray, optional
          An array of shape ``(N, K)`` of model-predicted class probabilities,
          ``P(label=k|x)``.
        :param confident_join: np.ndarray, optional
          An array of shape ``(K, K)`` representing the confident joint, the matrix used for identifying label issues, which
          estimates a confident subset of the joint distribution of the noisy and true labels
        :return:
        """
    def fit_transform(self, X: Dataset | pd.DataFrame, label_name: str, pred_probs: np.ndarray, metadata: Metadata, confident_joint: np.ndarray | None = None, n_jobs: int | None = None): ...
