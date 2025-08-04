import numpy as np

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
def get_confident_thresholds(labels: list | np.ndarray, pred_probs: np.ndarray) -> np.ndarray:
    '''Returns expected (average) "self-confidence" for each class. The
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
    '''
def get_normalized_entropy(pred_probs: np.ndarray) -> np.ndarray:
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
