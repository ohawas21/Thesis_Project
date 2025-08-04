from _typeshed import Incomplete

class InverseGaussCDF:
    name: str
    eps: Incomplete
    estimator: Incomplete
    def __init__(self, eps: float = 1e-05) -> None:
        """Inverse Gaussian CDF Transformation.

        Args:
            eps (float): epsilon to clip the CDF to [eps, 1-eps]
        """
    def forward(self, X): ...
    def inverse(self, Z): ...
    def gradient(self, X): ...
