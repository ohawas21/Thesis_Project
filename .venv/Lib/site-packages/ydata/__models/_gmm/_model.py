import logging
from os import getenv

from numpy.random import shuffle
from scipy.optimize import minimize_scalar
from sklearn.mixture import GaussianMixture

from ydata.synthesizers.base_synthesizer import BaseSynthesizer
from ydata.synthesizers.logger import synthlogger_config
from ydata.utils.misc import log_time_factory

logger = synthlogger_config(verbose=getenv(
    "VERBOSE", "false").lower() == "true")


class GMMSynthesizer(BaseSynthesizer):
    __name__ = "GMM"

    def __init__(self, *, max_components: int = 5):
        self.max_components = max_components

    @log_time_factory(logger)
    def fit(self, X, y=None, **kwargs):
        verbose = 2 if logger.getEffectiveLevel() == logging.DEBUG else 0
        cache = {}

        logger.info("Start fitting GMM synth.")

        def _fun(x: float) -> float:
            n_components = int(x)

            if n_components in cache:
                return cache[n_components]

            gmm = GaussianMixture(
                n_components=n_components, verbose=verbose, verbose_interval=1
            )

            logger.debug(
                f"Training synthesizer with n_components={n_components}")
            gmm.fit(X, y)

            if gmm.converged_:
                bic = gmm.bic(X)
            else:
                bic = 0

            cache[n_components] = bic

            return bic

        res = minimize_scalar(
            fun=_fun,
            method="Bounded",
            bounds=(1, self.max_components),
            options={"xatol": 0.5},
        )

        if not res.success:
            raise ValueError(res.message)

        logger.debug(f"Best n_components: {int(res.x)}")

        self.model_ = GaussianMixture(n_components=int(res.x), verbose=verbose).fit(
            X, y
        )

        logger.info("End fitting GMM model. Synth was trained succesfully.")

        return self

    @log_time_factory(logger)
    def sample(self, n_samples: int = 1):
        X, _ = self.model_.sample(n_samples)

        shuffle(X)

        return X
