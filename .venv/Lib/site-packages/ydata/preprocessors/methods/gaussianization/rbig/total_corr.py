import numpy as np
from tqdm import trange

from ydata.preprocessors.methods.gaussianization.rbig.entropy import entropy_marginal
from ydata.preprocessors.methods.gaussianization.rbig.invcdf import InverseGaussCDF
from ydata.preprocessors.methods.gaussianization.rbig.rotation import rotation_factory
from ydata.preprocessors.methods.gaussianization.rbig.uniform import MarginalHistogramUniformization, auto_bins


def information_reduction(X, Y, bins="auto", tol_dimensions=None, correction=True):
    """Computes the total correlation reduction after a linear transformation.

        Y = X * W
        II = I(X) - I(Y)

    Args:
        X (np.ndarray): original dataset
        Y (np.ndarray): dataset after the linear transformation
        bins (Union[int, str]): histogram estimation
        tol_dimensions (float): tolerance on the minimum multi-information difference
        correction (bool): If true, the estimation is corrected using Miller-Madow bias correction

    Returns:
        float: Information reduction


    Author: Valero Laparra
            Juan Emmanuel Johnson
    """
    # check that number of samples and dimensions are equal
    err_msg = "Number of samples for x and y should be equal."
    np.testing.assert_equal(X.shape, Y.shape, err_msg=err_msg)

    n_samples, n_dimensions = X.shape

    # minimum multi-information heuristic
    if tol_dimensions is None or 0:
        xxx = np.logspace(2, 8, 7)
        yyy = [0.1571, 0.0468, 0.0145, 0.0046, 0.0014, 0.0001, 0.00001]
        tol_dimensions = np.interp(n_samples, xxx, yyy)

    # preallocate data
    hx = np.zeros(n_dimensions)
    hy = np.zeros(n_dimensions)

    # calculate the marginal entropy
    if isinstance(bins, str) and bins == "auto":
        bins = auto_bins(X)
    hx = entropy_marginal(X, bins=bins, correction=correction)
    hy = entropy_marginal(Y, bins=bins, correction=correction)

    # Information content
    I = np.sum(hy) - np.sum(hx)  # noqa: E741
    II = np.sqrt(np.sum((hy - hx) ** 2))

    p = 0.25
    if II < np.sqrt(n_dimensions * p * tol_dimensions**2) or I < 0:
        I = 0  # noqa: E741

    return I


def rbig_total_corr(
    X: np.ndarray,
    bins: str = "auto",
    alpha: float = 1e-10,
    bound_ext: float = 0.3,
    eps: float = 1e-10,
    rotation: str = "PCA",
    zero_tolerance: int = 60,
    max_layers: int = 1_000,
    progress_bar: bool = False,
) -> float:
    """Calculate the total correlation of a RBIG model.

    Args:
        X (np.ndarray): dataset
        bins (Union[int, str]): histogram estimation
        alpha (float): regularization on the histogram
        bound_ext (float): tolerance factor on the histogram domain
        eps (float): epsilon on the domain to consider [ep, 1- eps] for the Inverse Gaussian CDF
        rotation (str): Type of rotation to use in the flow
        zero_tolerance (int): minimum number of transformations in the flow
        max_layers (int): maximum number of transformations in the flow
        progress_bar (bool): output progress bar in stdout

    Returns:
        float: total correlation
    """

    Z = X.copy()
    info_losses = []

    # initialize loss
    with trange(max_layers, disable=not progress_bar) as pbar:
        for ilayer in pbar:
            X_before = Z.copy()
            # Marginal Uniformization
            ibijector = MarginalHistogramUniformization(
                X=Z, bound_ext=bound_ext, bins=bins, alpha=alpha
            )

            Z = ibijector.forward(Z)

            # Inverse Gauss CDF
            ibijector = InverseGaussCDF(eps=eps)
            Z = ibijector.forward(Z)

            # Rotation
            ibijector = rotation_factory(rotation, Z)

            Z = ibijector.forward(Z)
            info_red = information_reduction(X=X_before, Y=Z, bins=bins)
            info_losses.append(info_red)

            if ilayer > zero_tolerance:
                if np.sum(np.abs(info_losses[-zero_tolerance:])) == 0:
                    info_losses = info_losses[:-zero_tolerance]
                    pbar.set_description(
                        f"Completed! (Total Information Reduction: {np.sum(info_losses):.4f})"
                    )
                    break

            pbar.set_description(f"Information Reduction: {info_red:.2e}")

    return np.array(info_losses).sum()
