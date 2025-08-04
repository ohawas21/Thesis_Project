import numpy as np
from scipy.stats import multivariate_normal
from tqdm import trange

from ydata.preprocessors.methods.gaussianization.rbig.base import FlowModel
from ydata.preprocessors.methods.gaussianization.rbig.config import RBIGConfig
from ydata.preprocessors.methods.gaussianization.rbig.invcdf import InverseGaussCDF
from ydata.preprocessors.methods.gaussianization.rbig.rotation import rotation_factory
from ydata.preprocessors.methods.gaussianization.rbig.total_corr import information_reduction
from ydata.preprocessors.methods.gaussianization.rbig.uniform import (MarginalHistogramUniformization,
                                                                      MarginalKDEUniformization)


def train_rbig_info_loss(
    X: np.ndarray, config: RBIGConfig, progress_bar: bool = False
) -> FlowModel:
    """Train and return a RBIG model.

    Args:
        X (np.ndarray): dataset
        config (RBIGConfig): Model config
        progress_bar (bool): output progress bar in stdout

    Returns:
        FlowModel: RBIG model
    """
    Z = X.copy()
    transformations = []
    info_losses = []

    # initialize loss
    with trange(config.max_layers, disable=not progress_bar) as pbar:
        for ilayer in pbar:
            X_before = Z.copy()
            # Marginal Uniformization
            if config.uniformizer == "hist":
                # if not config.domain_hint:
                #     domain_hint = [(iX.min(), iX.max()) for iX in X.T]
                ibijector = MarginalHistogramUniformization(
                    X=Z,
                    bound_ext=config.bound_ext,
                    bins=config.bins,
                    alpha=config.alpha,
                    domain_hint=config.domain_hint,
                    privacy_noise=None
                )
            elif config.uniformizer == "kde":
                ibijector = MarginalKDEUniformization(
                    X=Z, bound_ext=config.bound_ext, fft=True
                )

            transformations.append(ibijector)
            Z = ibijector.forward(Z)

            # Inverse Gauss CDF
            ibijector = InverseGaussCDF(eps=config.eps)
            transformations.append(ibijector)
            Z = ibijector.forward(Z)

            # Rotation
            ibijector = rotation_factory(
                config.rotation, Z, config.random_state, config.max_iter
            )

            transformations.append(ibijector)
            Z = ibijector.forward(Z)

            # The calculation fails in a few random scenarios.
            # When it fails, the information reduction is ignored.
            try:
                info_red = information_reduction(
                    X=X_before,
                    Y=Z,
                    bins=config.bins,
                )
                info_losses.append(info_red)
            except Exception:
                info_red = None

            if ilayer > config.zero_tolerance:
                if np.sum(np.abs(info_losses[-config.zero_tolerance:])) == 0:
                    info_losses = info_losses[: -config.zero_tolerance]
                    transformations = transformations[: -
                                                      3 * config.zero_tolerance]
                    pbar.set_description(
                        f"Completed! (Total Information Reduction: {np.sum(info_losses):.4f})"
                    )
                    break

            if info_red is not None:
                pbar.set_description(f"Information Reduction: {info_red:.2e}")

    base_dist = multivariate_normal(
        mean=np.zeros(X.shape[1]), cov=np.ones(X.shape[1]))

    gf_model = FlowModel(transformations, base_dist)
    gf_model.info_loss = np.array(info_losses)

    return gf_model
