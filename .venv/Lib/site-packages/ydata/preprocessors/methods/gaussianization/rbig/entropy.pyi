import numpy as np

def entropy_univariate(X: np.ndarray, bins: int | str = 'auto', correction: bool = True) -> np.ndarray:
    """Calculate the entropy of a univariate dataset.

    Args:
        X (np.ndarray): dataset
        bins (Union[int, str]): histogram estimation
        correction (bool): If true, the estimation is corrected using Miller-Madow bias correction

    Returns:
        np.ndarray: entry
    """
def entropy_marginal(X: np.ndarray, bins: int | str = 'auto', correction: bool = True) -> np.ndarray:
    """Calculate the marginal entropy of a multivariate dataset.

    Args:
        X (np.ndarray): dataset
        bins (Union[int, str]): histogram estimation
        correction (bool): If true, the estimation is corrected using Miller-Madow bias correction

    Returns:
        np.ndarray: entry
    """
def entropy_rbig(X: np.ndarray, bins: int | str = 'auto', alpha: float = 1e-10, bound_ext: float = 0.3, eps: float = 1e-10, rotation: str = 'PCA', zero_tolerance: int = 60, max_layers: int = 1000, progress_bar: bool = False):
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
