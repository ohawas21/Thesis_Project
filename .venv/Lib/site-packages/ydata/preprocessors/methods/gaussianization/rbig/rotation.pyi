import numpy as np
from _typeshed import Incomplete

def rotation_factory(rotation: str, X: np.ndarray, random_state: int = 42, max_iter: Incomplete | None = None): ...

class ICARotation:
    name: str
    estimator: Incomplete
    def __init__(self, X: np.ndarray, random_state: int = 123, max_iter: int = 10, **kwargs) -> None:
        """ICA rotation based on sklearn.decomposition.FastICA.

        Args:
            X (np.ndarray): dataset
            random_state (int): random state for the operator
            max_iter (int): maximum number of iterations
        """
    def forward(self, X): ...
    def inverse(self, Z): ...
    def gradient(self, X): ...

class PCARotation:
    name: str
    estimator: Incomplete
    def __init__(self, X: np.ndarray, **kwargs) -> None:
        """PCA rotation based on sklearn.decomposition.PCA.

        Args:
            X (np.ndarray): dataset
        """
    def forward(self, X): ...
    def inverse(self, Z): ...
    def gradient(self, X): ...

class RandomRotation:
    name: str
    rand_ortho_matrix: Incomplete
    def __init__(self, X: np.ndarray, **kwargs) -> None:
        """Random rotation based on scipy random orthogonal matrix.

        Args:
            X (np.ndarray): dataset
        """
    def forward(self, X): ...
    def inverse(self, Z): ...
    def gradient(self, X): ...
