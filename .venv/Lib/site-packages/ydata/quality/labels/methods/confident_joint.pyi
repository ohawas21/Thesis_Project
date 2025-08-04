import numpy as np

def round_preserving_row_totals(confident_joint: np.ndarray) -> np.ndarray:
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
def calibrate_confident_joint(confident_joint: np.ndarray, labels: list | np.ndarray, label_counts: np.ndarray) -> np.ndarray:
    """Calibrates any confident joint estimate ``P(label=i, true_label=j)`` such that
    ``np.sum(cj) == len(labels)`` and ``np.sum(cj, axis = 1) == np.bincount(labels)``.

    In other words, this function forces the confident joint to have the
    true noisy prior ``p(labels)`` (summed over columns for each row) and also
    forces the confident joint to add up to the total number of examples.

    This method makes the confident joint a valid counts estimate
    of the actual joint of noisy and true labels.

    Parameters
    ----------
    confident_joint : np.ndarray
      An array of shape ``(K, K)`` representing the confident joint, the matrix used for identifying label issues, which
      estimates a confident subset of the joint distribution of the noisy and true labels, ``P_{noisy label, true label}``.
      Entry ``(j, k)`` in the matrix is the number of examples confidently counted into the pair of ``(noisy label=j, true label=k)`` classes.
      The `confident_joint` can be computed using `~cleanlab.count.compute_confident_joint`.
      If not provided, it is computed from the given (noisy) `labels` and `pred_probs`.

    labels : np.ndarray or list
      Given class labels for each example in the dataset, some of which may be erroneous,

    Returns
    -------
    calibrated_cj : np.ndarray
      An array of shape ``(K, K)`` representing a valid estimate of the joint *counts* of noisy and true labels (if `multi_label` is False).
      If `multi_label` is True, the returned `calibrated_cj` is instead an one-vs-rest array of shape ``(K, 2, 2)``,
      where for class `c`: entry ``(c, 0, 0)`` in this one-vs-rest  array is the number of examples whose noisy label contains `c` confidently identified as truly belonging to class `c` as well.
      Entry ``(c, 1, 0)`` in this one-vs-rest  array is the number of examples whose noisy label contains `c` confidently identified as not actually belonging to class `c`.
      Entry ``(c, 0, 1)`` in this one-vs-rest array is the number of examples whose noisy label does not contain `c` confidently identified as truly belonging to class `c`.
      Entry ``(c, 1, 1)`` in this one-vs-rest array is the number of examples whose noisy label does not contain `c` confidently identified as actually not belonging to class `c` as well.
    """
def compute_confident_joint(labels, label_counts: np.ndarray, pred_probs: np.ndarray, *, thresholds: np.ndarray | list | None = None, calibrate: bool = True, return_indices_of_off_diagonals: bool = False):
    """Estimates the confident counts of latent true vs observed noisy labels
    for the examples in our dataset. This array of shape ``(K, K)`` is called
    the **confident joint** and contains counts of examples in every class,
    confidently labeled as every other class. These counts may subsequently be
    used to estimate the joint distribution of true and noisy labels (by
    normalizing them to frequencies).

    Important: this function assumes that `pred_probs` are out-of-sample
    holdout probabilities. This can be :ref:`done with cross validation <pred_probs_cross_val>`. If
    the probabilities are not computed out-of-sample, overfitting may occur.

    :param labels:np.ndarray or list
      Given class labels for each example in the dataset, some of which may be erroneous
    :param pred_probs: np.ndarray
      Model-predicted class probabilities for each example in the dataset
    :param thresholds: rray_like, optional
      An array of shape ``(K, 1)`` or ``(K,)`` of per-class threshold
      probabilities, used to determine the cutoff probability necessary to
      consider an example as a given class label (see `Northcutt et al.,
      2021 <https://jair.org/index.php/jair/article/view/12125>`_, Section
      3.1, Equation 2).).
      This is for advanced used cases. If it is not specified it will be computed automatically.
      If an example has a predicted probability
      greater than this threshold, it is counted as having true_label =
      k. This is not used for pruning/filtering, only for estimating the
      noise rates using confident counts.
    :param calibrate: bool, default=True
        Calibrates confident joint estimate ``P(label=i, true_label=j)`` such that
        ``np.sum(cj) == len(labels)`` and ``np.sum(cj, axis = 1) == np.bincount(labels)``.
        When ``calibrate=True``, this method returns an estimate of
        the latent true joint counts of noisy and true labels.

    :param return_indices_of_off_diagonals: bool, optional
      If ``True``, returns indices of examples that were counted in off-diagonals
      of confident joint as a baseline proxy for the label issues. This
      sometimes works as well as ``filter.find_label_issues(confident_joint)``.

    :return: confident_joint_counts : np.ndarray
      An array of shape ``(K, K)`` representing counts of examples
      for which we are confident about their given and true label (if `multi_label` is False).
      If `multi_label` is True,
      this array instead has shape ``(K, 2, 2)`` representing a one-vs-rest format for the  confident joint, where for each class `c`:
      Entry ``(c, 0, 0)`` in this one-vs-rest array is the number of examples whose noisy label contains `c` confidently identified as truly belonging to class `c` as well.
      Entry ``(c, 1, 0)`` in this one-vs-rest array is the number of examples whose noisy label contains `c` confidently identified as not actually belonging to class `c`.
      Entry ``(c, 0, 1)`` in this one-vs-rest array is the number of examples whose noisy label does not contain `c` confidently identified as truly belonging to class `c`.
      Entry ``(c, 1, 1)`` in this one-vs-rest array is the number of examples whose noisy label does not contain `c` confidently identified as actually not belonging to class `c` as well.
    """
