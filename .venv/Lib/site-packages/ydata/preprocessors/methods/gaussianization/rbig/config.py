from dataclasses import dataclass
from typing import List, Optional, Union


@dataclass
class RBIGConfig:
    """RBIG model configuration.

    Args:
        uniformizer (str): type of uniformizer
        bins (Union[int, str]): histogram estimation
        alpha (float): regularization on the histogram
        bound_ext (float): tolerance factor on the histogram domain
        eps (float): epsilon on the domain to consider [ep, 1- eps] for the Inverse Gaussian CDF
        rotation (str): type of rotation to use in the flow
        zero_tolerance (int): minimum number of transformations in the flow
        max_layers (int): maximum number of transformations in the flow
        random_state (Optional[int]): random state for the different operators
        max_iter (int): maximum number of iterations for the rotation operator
        domain_hint (Optional[List[float]]): bounds on the marginal domains
    """

    uniformizer: str = "hist"
    bins: Union[int, str] = "auto"
    alpha: float = 1e-10
    bound_ext: float = 0.3
    eps: float = 1e-10
    rotation: str = "PCA"
    zero_tolerance: int = 60
    max_layers: int = 1_000
    random_state: Optional[int] = 123
    max_iter: int = 10
    domain_hint: Optional[List[float]] = None
