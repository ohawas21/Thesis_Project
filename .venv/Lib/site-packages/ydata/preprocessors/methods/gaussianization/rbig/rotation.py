import numpy as np
from scipy.stats import ortho_group
from sklearn.decomposition import PCA, FastICA


def rotation_factory(rotation: str, X: np.ndarray, random_state=42, max_iter=None):
    if rotation.lower() == "pca":
        ibijector = PCARotation(X=X, random_state=random_state)
    elif rotation.lower() == "random":
        ibijector = RandomRotation(X=X)
    elif rotation.lower() == "ica":
        ibijector = ICARotation(
            X=X, random_state=random_state, max_iter=max_iter)
    else:
        raise ValueError(f"Unrecognized rotation method: {rotation}")
    return ibijector


class ICARotation:
    name: str = "ica"

    def __init__(
        self, X: np.ndarray, random_state: int = 123, max_iter: int = 10, **kwargs
    ):
        """ICA rotation based on sklearn.decomposition.FastICA.

        Args:
            X (np.ndarray): dataset
            random_state (int): random state for the operator
            max_iter (int): maximum number of iterations
        """
        self.estimator = FastICA(
            whiten=False,
            random_state=random_state,
            max_iter=max_iter,
            n_components=None,
            **kwargs,
        ).fit(X)

    def forward(self, X):

        Z = self.estimator.transform(X)

        return Z

    def inverse(self, Z):
        X = self.estimator.inverse_transform(Z)

        return X

    def gradient(self, X):

        X_grad = np.zeros(X.shape[0])

        return X_grad


class PCARotation:
    name: str = "pca"

    def __init__(self, X: np.ndarray, **kwargs):
        """PCA rotation based on sklearn.decomposition.PCA.

        Args:
            X (np.ndarray): dataset
        """
        self.estimator = PCA().fit(X)
        # self.estimator = IncrementalPCA().fit(X)

    def forward(self, X):

        Z = self.estimator.transform(X)

        return Z

    def inverse(self, Z):
        X = self.estimator.inverse_transform(Z)

        return X

    def gradient(self, X):

        X_grad = np.zeros(X.shape[0])

        return X_grad


class RandomRotation:
    name: str = "random"

    def __init__(self, X: np.ndarray, **kwargs):
        """Random rotation based on scipy random orthogonal matrix.

        Args:
            X (np.ndarray): dataset
        """
        # create histogram object
        self.rand_ortho_matrix = ortho_group.rvs(X.shape[1])

    def forward(self, X):

        Z = X @ self.rand_ortho_matrix

        return Z

    def inverse(self, Z):
        X = Z @ self.rand_ortho_matrix.T

        return X

    def gradient(self, X):

        X_grad = np.zeros(X.shape[0])

        return X_grad
