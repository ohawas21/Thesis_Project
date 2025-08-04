import numpy as np
from _typeshed import Incomplete
from typing import NamedTuple

def auto_bins(data): ...

class MarginalHistogramUniformization:
    name: str
    estimators: Incomplete
    def __init__(self, X: np.ndarray, bins: int | str = 'auto', alpha: float = 1e-10, bound_ext: float = 0.1, domain_hint: list[float] | None = None, privacy_noise: float = None) -> None:
        """Uniformize marginal distributions using the marginal histograms of
        another dataset.

        Args:
            X (np.ndarray): dataset
            bins (Union[int, str]): histogram estimation
            alpha (float): regularization on the histogram
            bound_ext (float): tolerance factor on the histogram domain
            domain_hint (Optional[List[float]]): bounds on the marginal domains
            privacy_noise (float): differential privacy noise
        """
    def forward(self, X): ...
    def inverse(self, Z): ...
    def gradient(self, X): ...

class KDEParams(NamedTuple):
    support: np.ndarray
    pdf_est: np.ndarray
    cdf_est: np.ndarray

class MarginalKDEUniformization:
    name: str
    estimators: Incomplete
    def __init__(self, X: np.ndarray, grid_size: int = 50, n_quantiles: int = 1000, bound_ext: float = 0.1, fft: bool = True) -> None:
        """Uniformize marginal distributions using the Kernel Distribution of
        the marginal of another dataset.

        Args:
            X (np.ndarray): dataset
            bins (Union[int, str]): histogram estimation
            alpha (float): regularization on the histogram
            bound_ext (float): tolerance factor on the histogram domain
            domain_hint: Optional[List[float]]: Bounds on the marginal domains
        """
    def forward(self, X): ...
    def inverse(self, Z): ...
    def gradient(self, X): ...
