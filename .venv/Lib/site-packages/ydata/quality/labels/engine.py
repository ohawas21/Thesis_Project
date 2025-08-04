"""File to define the class for the Engine to identify potential inconsistent
labels."""
import multiprocessing
import sys
import warnings
from typing import List, Union

import numpy as np
import pandas as pd
import psutil
import tqdm
from sklearn.base import BaseEstimator, TransformerMixin
from typeguard import typechecked

from ydata.dataset import Dataset
from ydata.metadata import Metadata
from ydata.quality.labels.enums import LabelFilter
from ydata.quality.labels.methods.confident_joint import compute_confident_joint
from ydata.quality.labels.methods.prune import _keep_at_least_n_per_class
from ydata.quality.labels.methods.rank import RankedBy, order_label_issues
from ydata.quality.labels.utils import _reduce_issues, round_preserving_row_totals


def _set_n_jobs(n_jobs):
    # Setting the multiprocessing threads
    # Move this to a separate function
    if n_jobs is None:
        n_jobs = psutil.cpu_count(logical=False)
        if not n_jobs:
            # either psutil does not exist
            # or psutil can return None when physical cores cannot be determined
            # switch to logical cores
            n_jobs = multiprocessing.cpu_count()
    else:
        assert n_jobs >= 1, "Number of jobs must be equal or bigger than 1."

    return n_jobs


def get_unique_classes(labels, multi_label=None) -> set:
    """Returns the set of unique classes for both single-labeled and multi-
    labeled labels.

    If multi_label is set to None (default) this method will infer if
    multi_label is True or False based on the format of labels. This
    allows for a more general form of multiclass labels that looks like
    this: [1, [1,2], [0], [0, 1], 2, 1]
    """
    if multi_label is None:
        multi_label = any(isinstance(label, list) for label in labels)
    if multi_label:
        return set(label for grp in labels for label in list(grp))
    else:
        return set(labels)


def num_unique_classes(labels, multi_label=None) -> int:
    """Finds the number of unique classes for both single-labeled and multi-
    labeled labels.

    If multi_label is set to None (default) this method will infer if
    multi_label is True or False based on the format of labels. This
    allows for a more general form of multiclass labels that looks like
    this: [1, [1,2], [0], [0, 1], 2, 1]
    """
    return len(get_unique_classes(labels, multi_label))


def get_num_classes(labels=None,
                    pred_probs=None,
                    label_matrix=None,
                    multi_label=None) -> int:
    """Determines the number of classes based on information considered in a
    canonical ordering.

    label_matrix can be: noise_matrix, inverse_noise_matrix, confident_joint,
    or any other K x K matrix where K = number of classes.
    """
    if pred_probs is not None:  # pred_probs is number 1 source of truth
        return pred_probs.shape[1]

    if label_matrix is not None:  # matrix dimension is number 2 source of truth
        if label_matrix.shape[0] != label_matrix.shape[1]:
            raise ValueError(
                f"label matrix must be K x K, not {label_matrix.shape}")
        else:
            return label_matrix.shape[0]

    if labels is None:
        raise ValueError("Cannot determine number of classes from None input")

    return num_unique_classes(labels, multi_label=multi_label)


def _prune_by_count(args: list) -> np.ndarray:
    """multiprocessing Helper function for find_label_issues() that assumes
    globals and produces a mask for class k for each example by removing the
    example with noisy label k having *largest margin*, where.

    margin of example := prob of given label - max prob of non-given labels

    Parameters
    ----------
    k : int (between 0 and num classes - 1)
      The true_label class of interest.
    """

    k, min_examples_per_class, arrays = args
    if arrays is None:
        pred_probs = pred_probs_by_class[k]
        prune_count_matrix = prune_count_matrix_cols[k]
    else:
        pred_probs = arrays[0]
        prune_count_matrix = arrays[1]

    label_counts = pred_probs.shape[0]
    label_issues_mask = np.zeros(label_counts, dtype=bool)
    if label_counts <= min_examples_per_class:
        warnings.warn(
            f"May not flag all label issues in class: {k}, it has too few examples (see `min_examples_per_class` argument)"
        )
        return label_issues_mask

    K = pred_probs.shape[1]
    if K < 1:
        raise ValueError("Must have at least 1 class.")
    for j in range(K):
        num2prune = prune_count_matrix[j]
        # Only prune for noise rates, not diagonal entries
        if k != j and num2prune > 0:
            # num2prune's largest p(true class k) - p(noisy class k)
            # for x with true label j
            margin = pred_probs[:, j] - pred_probs[:, k]
            order = np.argsort(-margin)
            label_issues_mask[order[:num2prune]] = True
    return label_issues_mask


@typechecked
class FindInconsistentLabelsEngine(BaseEstimator, TransformerMixin):

    def __init__(self, filter_type: Union[str, LabelFilter] = LabelFilter.PRUNE,
                 frac_noise: float = 1.0,
                 num_to_remove_per_class: List[int] | None = None,
                 min_examples_per_class: int = 1,
                 indices_ranked_by: Union[str, RankedBy] = RankedBy.DEFAULT,):
        """Identifies potentially bad labels in a classification dataset using
        different filtering methods like pruning and confident learning.

        This class return a boolean mask for the entire dataset where ``True`` represents
        an example identified with a label issue and ``False`` represents an example that seems correctly labeled.

        Instead of a mask, you can obtain indices of the examples with label issues in your dataset
        (sorted by issue severity) by specifying the `return_indices_ranked_by` argument.
        This determines which label quality score is used to quantify severity,
        and is useful to view only the top-`J` most severe issues in your dataset.

        :param filter_type: FilterType ('prune_by_noise_rate', 'confident_learning'), default is set to FilterType.PRUNE
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
        """
        self.filter_type = LabelFilter(filter_type)
        self.ranked_by = RankedBy.from_value(indices_ranked_by) if isinstance(
            indices_ranked_by, RankedBy) is False else indices_ranked_by

        if (filter_type == LabelFilter.CONFIDENT_LEARNING) and \
                (frac_noise != 1.0 or num_to_remove_per_class is not None):
            warnings.warn(
                "The parameters frac_noise and num_to_remove_per_class will be ignored as"
                "they are only supported for filter_type 'prune'. "
            )

        self.frac_noise = frac_noise
        self.num_to_remove_per_class = num_to_remove_per_class
        self.min_examples_per_class = min_examples_per_class

    def _set_confident_joint(self, confident_join):
        if self.filter_type == LabelFilter.CONFIDENT_LEARNING and isinstance(confident_join, np.ndarray):
            warnings.warn(
                "The given confident_join matrix will be ignored. When filter_type 'confident_learning' is "
                "selected confident_join will be re-estimated from the provided table of values."
            )
            confident_join = None

        return confident_join

    def fit(self,
            X: Union[Dataset, pd.DataFrame],
            metadata: Metadata,
            confident_joint: np.ndarray | None = None):

        # validate the cols here
        assert list(X.columns) == list(
            metadata.columns.keys()), "The provided metadata must belong to the provided dataset."

        self.confident_joint = self._set_confident_joint(
            confident_join=confident_joint)

        self.n_classes = metadata.summary['cardinality']['Class']

        # Returns a boolean to identify whether the dataset is large
        self.big_dataset = self.n_classes * len(X) > 1e8

        # replace with the metadata
        self.label_counts = np.asarray(
            metadata.summary['value_counts']['Class'])

        return self

    def transform(self, X: Union[Dataset, pd.DataFrame], label_name: str, pred_probs: np.ndarray, n_jobs: int | None = None):
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
        assert label_name in X.columns, "The provided label name must exist in the dataset."

        # Setting the number of jobs for multiprocessing
        n_jobs = _set_n_jobs(n_jobs=n_jobs)

        if isinstance(X, Dataset):
            labels = np.asarray(X[label_name].to_pandas())
            labels = labels.reshape(labels.shape[0])
        else:
            labels = np.asarray(X[label_name])

        if self.confident_joint is None or self.filter_type == LabelFilter.CONFIDENT_LEARNING:
            self.confident_joint, self.cl_error_indices = compute_confident_joint(labels=labels,
                                                                                  label_counts=self.label_counts,
                                                                                  pred_probs=pred_probs,
                                                                                  return_indices_of_off_diagonals=True)

        if self.filter_type == LabelFilter.PRUNE:
            prune_count_matrix = _keep_at_least_n_per_class(prune_count_matrix=self.confident_joint.T,
                                                            n=self.min_examples_per_class,
                                                            frac_noise=self.frac_noise)

            if self.num_to_remove_per_class is not None:
                # Estimate joint probability distribution over label issues
                psy = prune_count_matrix / np.sum(prune_count_matrix, axis=1)
                noise_per_s = psy.sum(axis=1) - psy.diagonal()
                # Calibrate labels.t. noise rates sum to num_to_remove_per_class
                tmp = (psy.T * self.naum_to_remove_per_class / noise_per_s).T
                np.fill_diagonal(tmp, self.label_counts -
                                 self.num_to_remove_per_class)
                prune_count_matrix = round_preserving_row_totals(tmp)

            # Prepare multiprocessing
            chunksize = max(1, self.n_classes // n_jobs)
            if n_jobs == 1:
                global pred_probs_by_class, prune_count_matrix_cols
                pred_probs_by_class = {
                    k: pred_probs[labels == k] for k in range(self.n_classes)}
                prune_count_matrix_cols = {
                    k: prune_count_matrix[:, k] for k in range(self.n_classes)}
                args = [[k, self.min_examples_per_class, None]
                        for k in range(self.n_classes)]
            else:
                args = [
                    [k, self.min_examples_per_class, [
                        pred_probs[labels == k], prune_count_matrix[:, k]]]
                    for k in range(self.n_classes)
                ]

        if self.filter_type == LabelFilter.PRUNE:
            if n_jobs > 1:
                with multiprocessing.Pool(n_jobs) as p:
                    sys.stdout.flush()

                    if self.big_dataset:  # add here the tqdm
                        label_issues_masks_per_class = list(
                            tqdm.tqdm(p.imap(_prune_by_count, args,
                                      chunksize=chunksize), total=self.n_classes)
                        )
                    else:
                        label_issues_masks_per_class = p.map(
                            _prune_by_count, args, chunksize=chunksize)
            else:
                label_issues_masks_per_class = [
                    _prune_by_count(arg) for arg in args]

            label_issues_mask = np.zeros(len(labels), dtype=bool)
            for k, mask in enumerate(label_issues_masks_per_class):
                if len(mask) > 1:
                    label_issues_mask[labels == k] = mask

        if self.filter_type == LabelFilter.CONFIDENT_LEARNING:  # I might have to see whether this is right
            label_issues_mask = np.zeros(len(labels), dtype=bool)
            label_issues_mask[self.cl_error_indices] = True

        mask = _reduce_issues(pred_probs=pred_probs, labels=labels)
        label_issues_mask[mask] = False

        if self.ranked_by != RankedBy.DEFAULT:
            ordered_index = order_label_issues(label_issues_mask=label_issues_mask,
                                               labels=labels,
                                               pred_probs=pred_probs,
                                               rank_by=self.ranked_by,)
            return ordered_index
        return label_issues_mask

    def fit_transform(self, X: Union[Dataset, pd.DataFrame],
                      label_name: str,
                      pred_probs: np.ndarray,
                      metadata: Metadata,
                      confident_joint: np.ndarray | None = None,
                      n_jobs: int | None = None,):

        self.fit(X=X,
                 metadata=metadata,
                 confident_joint=confident_joint)

        return self.transform(X=X, label_name=label_name, pred_probs=pred_probs, n_jobs=n_jobs)
