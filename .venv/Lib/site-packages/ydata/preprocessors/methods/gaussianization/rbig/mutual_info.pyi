from _typeshed import Incomplete

class MutualInfoRBIG:
    bins: Incomplete
    alpha: Incomplete
    bound_ext: Incomplete
    eps: Incomplete
    rotation: Incomplete
    zero_tolerance: Incomplete
    max_layers: Incomplete
    def __init__(self, bins: int | str = 'auto', alpha: float = 1e-10, bound_ext: float = 0.3, eps: float = 1e-10, rotation: str = 'PCA', zero_tolerance: int = 60, max_layers: int = 1000) -> None:
        """Calculate the mutual information between two datasets based on RBIG
        transformation.

        Args:
            bins (Union[int, str]): histogram estimation
            alpha (float): regularization on the histogram
            bound_ext (float): tolerance factor on the histogram domain
            eps (float): epsilon on the domain to consider [ep, 1- eps] for the Inverse Gaussian CDF
            rotation (str): Type of rotation to use in the flow
            zero_tolerance (int): minimum number of transformations in the flow
            max_layers (int): maximum number of transformations in the flow

        Returns:
            FlowModel: RBIG model
        """
    rbig_model_X: Incomplete
    rbig_model_Y: Incomplete
    rbig_model_XY: Incomplete
    def fit(self, X, Y): ...
    def mutual_info(self): ...
