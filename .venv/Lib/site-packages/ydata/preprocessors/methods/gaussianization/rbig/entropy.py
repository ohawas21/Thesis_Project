from typing import Union

import numpy as np
from scipy.stats import rv_histogram


def entropy_univariate(
    X: np.ndarray, bins: Union[int, str] = "auto", correction: bool = True
) -> np.ndarray:
    """Calculate the entropy of a univariate dataset.

    Args:
        X (np.ndarray): dataset
        bins (Union[int, str]): histogram estimation
        correction (bool): If true, the estimation is corrected using Miller-Madow bias correction

    Returns:
        np.ndarray: entry
    """
    hist = np.histogram(X, bins=bins, range=(X.min(), X.max()))

    # Calculate differential entropy
    H = rv_histogram(hist).entropy()

    # Miller-Madow correction
    # See: Miller, G. 1955. Note on the bias of information estimates. Info. Theory Psychol. Prob. Methods II-B:95-100.
    if correction:
        H += 0.5 * (np.sum(hist[0] > 0) - 1) / np.sum(hist[0])

    return H


def entropy_marginal(
    X: np.ndarray, bins: Union[int, str] = "auto", correction: bool = True
) -> np.ndarray:
    """Calculate the marginal entropy of a multivariate dataset.

    Args:
        X (np.ndarray): dataset
        bins (Union[int, str]): histogram estimation
        correction (bool): If true, the estimation is corrected using Miller-Madow bias correction

    Returns:
        np.ndarray: entry
    """
    d_dimensions = X.shape[1]

    H = np.zeros(d_dimensions)

    for idim, i_data in enumerate(X.T):
        H[idim] = entropy_univariate(i_data, bins=bins, correction=correction)

    return H


def entropy_rbig(
    X: np.ndarray,
    bins: (Union[int, str]) = "auto",
    alpha: float = 1e-10,
    bound_ext: float = 0.3,
    eps: float = 1e-10,
    rotation: str = "PCA",
    zero_tolerance: int = 60,
    max_layers: int = 1_000,
    progress_bar: bool = False,
):
    """Calculate total entropy of a RBIG model defined as the sum of marginal
    entropies minus the total correlation.

    Args:
        X (np.ndarray): dataset
        bins (Union[int, str]): histogram estimation
        alpha (float):
        bound_ext (float):
        eps (float): epsilon on the domain to consider [ep, 1- eps] for the Inverse Gaussian CDF
        rotation (str): type of rotation to use in the flow
        zero_tolerance (int): minimum number of operator in the flow
        max_layers (int): maximum number of transformations in the flow
        progress_bar (bool): output progress bar in stdout

    Returns:
        np.ndarray: entry
    """
    # Avoid circular import
    from ydata.preprocesssors.methods.gaussianization.rbig.total_corr import rbig_total_corr

    # total correlation using RBIG, TC(X)
    tc_rbig = rbig_total_corr(
        X=X,
        bins=bins,
        alpha=alpha,
        bound_ext=bound_ext,
        eps=eps,
        rotation=rotation,
        zero_tolerance=zero_tolerance,
        max_layers=max_layers,
        progress_bar=progress_bar,
    )

    # marginal entropy using rbig, H(X)
    Hx = entropy_marginal(X)

    # Multivariate entropy, H(X) - TC(X)
    return Hx.sum() - tc_rbig
