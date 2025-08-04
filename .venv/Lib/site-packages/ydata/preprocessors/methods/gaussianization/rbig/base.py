from typing import List

import numpy as np
from scipy.stats import norm
from scipy.stats._multivariate import multi_rv_generic


class Bijector:
    """Base class for a Bijector Mixin.

    A bijector is by definition fully invertible mapping with a
    gradient.
    """

    def forward(self, X):
        raise NotImplementedError

    def inverse(self, X):
        raise NotImplementedError

    def gradient(self, X):
        raise NotImplementedError


class CompositeBijector(Bijector):
    """A class to compose bijectors.

    It is itself a Bijector by definition of composition.
    """

    def __init__(self, bijectors: List[Bijector]):
        """Composite Bijector initialization.

        Args:
            bijectors (List[Bijector]): Ordered list of bijector to compose
        """
        self.bijectors = bijectors

    def forward(self, X):
        Z = X.copy()
        for ibijector in self.bijectors:
            Z = ibijector.forward(Z)
        return Z

    def inverse(self, Z):
        X = Z.copy()
        for ibijector in reversed(self.bijectors):
            X = ibijector.inverse(X)

        return X

    def gradient(self, X):
        Z = X.copy()
        X_grad = np.zeros_like(X).sum(axis=-1)
        for ibijector in self.bijectors:
            X_grad += ibijector.gradient(Z)
            Z = ibijector.forward(Z)
        return X_grad


class FlowModel(CompositeBijector):
    """Base Flow Model class.

    A flow model is a generative model using an explicit likelihood
    function defined by the composition of many change-of-variable
    bijective transformations to transform a distribution into another.
    """

    def __init__(self, bijectors: List[Bijector], base_dist):
        """Flow Model initialization.

        Args:
            bijectors (List[Bijector]): Ordered list of bijector to compose
            base_dist (multi_rv_generic): Initial distribution to sample
        """
        super().__init__(bijectors)
        self.base_dist = base_dist
        self.info_loss = None

    def sample(self, n_samples: multi_rv_generic = 10):
        pz_samples = self.base_dist.rvs(size=n_samples)
        X = self.inverse(pz_samples)
        return X

    def predict_proba(self, X):
        # forward tranformation
        Z = self.forward(X)
        pz = norm.logpdf(Z).sum(axis=-1)
        # gradient transformation
        X_ldj = self.gradient(X)
        return np.exp(pz + X_ldj)

    def score_samples(self, X):
        prob = self.predict_proba(X)
        return -np.mean(np.log(prob))
