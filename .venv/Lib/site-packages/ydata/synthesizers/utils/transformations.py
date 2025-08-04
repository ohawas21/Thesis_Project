from __future__ import absolute_import, division, print_function

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import PowerTransformer

from .checks import check_inputs


def skewness(col, data):
    """
    Parameters
    ----------
    col Name of the column to be transformed
    data series with the column values
    Returns A transformer function to correct skewness
    -------
    """
    tlog = PowerTransformer(method="yeo-johnson")
    data = tlog.fit_transform(data.values.reshape(-1, 1))

    result = {"model": tlog, "type": "skewness", "output_dimensions": 1}
    return result, data


class MultiModalNumberTransformer:
    r"""Reversible transform for multimodal data.
    To effectively sample values from a multimodal distribution, we cluster values of a
    numerical variable using a `skelarn.mixture.GaussianMixture`_ model (GMM).
    Args:
        num_modes(int): Number of modes on given data.
    Attributes:
        num_modes(int): Number of components in the `skelarn.mixture.GaussianMixture`_ model.
    .. _skelarn.mixture.GaussianMixture: https://scikit-learn.org/stable/modules/generated/
        sklearn.mixture.GaussianMixture.html
    """

    def __init__(self, num_modes=5):
        """Initialize instance."""
        self.num_modes = num_modes
        self.model = None
        self.means = None
        self.stds = None

    def fit(self, data):
        # Fazer aqui o step fit
        self.model = GaussianMixture(self.num_modes)
        self.model.fit(data)
        self.means = self.model.means_.reshape((1, self.num_modes))
        self.stds = np.sqrt(self.model.covariances_).reshape(
            (1, self.num_modes))

        result = {
            "model": self.model,
            "output_info": [(1, "tanh"), (self.num_modes, "softmax")],
            "output_dimensions": 1 + self.num_modes,
            "means": self.means,
            "stds": self.stds,
            "type": "GaussianMixture",
        }
        return result

    @check_inputs
    def transform(self, data):
        """Cluster values using a `skelarn.mixture.GaussianMixture`_ model.
        Args:
            data(numpy.ndarray): Values to cluster in array of shape (n,1).
        Returns:
            tuple[numpy.ndarray, numpy.ndarray, list, list]: Tuple containg the features,
            probabilities, averages and stds of the given data.
        .. _skelarn.mixture.GaussianMixture: https://scikit-learn.org/stable/modules/generated/
            sklearn.mixture.GaussianMixture.html
        """
        model = self.model.fit(data)

        features = (data - self.means) / (2 * self.stds)
        probs = model.predict_proba(data)
        argmax = np.argmax(probs, axis=1)
        idx = np.arange(len(features))
        features = features[idx, argmax].reshape([-1, 1])

        features = np.clip(features, -0.99, 0.99)

        return features, probs, list(self.means.flat), list(self.stds.flat)

    def inverse_transform(self, data):
        """Reverse the clustering of values.

        Args:
            data(numpy.ndarray): Transformed data to restore.
            info(dict): Metadata.
        Returns:
           numpy.ndarray: Values in the original space.
        """
        features = data[:, 0]
        probs = data[:, 1:]
        p_argmax = np.argmax(probs, axis=1)

        mean = np.asarray(self.means)
        std = np.asarray(self.stds)

        select_mean = mean[p_argmax]
        select_std = std[p_argmax]

        return features * 2 * select_std + select_mean
