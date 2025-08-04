import numpy as np
from _typeshed import Incomplete

def information_reduction(X, Y, bins: str = 'auto', tol_dimensions: Incomplete | None = None, correction: bool = True):
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
def rbig_total_corr(X: np.ndarray, bins: str = 'auto', alpha: float = 1e-10, bound_ext: float = 0.3, eps: float = 1e-10, rotation: str = 'PCA', zero_tolerance: int = 60, max_layers: int = 1000, progress_bar: bool = False) -> float:
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
