from typing import List, NamedTuple, Optional, Union

import numpy as np
import statsmodels.api as sm
from scipy.stats import rv_histogram
from statsmodels.distributions.empirical_distribution import ECDF


def auto_bins(data):
    a_unsorted = np.array(data)
    left_cap, right_cap = a_unsorted.min(), a_unsorted.max()
    a = np.sort(a_unsorted) - left_cap
    fd = np.lib.histograms._hist_bin_fd(a, range)
    left_edges = a // fd * fd
    right_edges = left_edges + fd
    new_bins = np.unique(np.concatenate((left_edges, right_edges))) + left_cap
    return np.append(new_bins, right_cap + fd)


class MarginalHistogramUniformization:
    name: str = "marghistuni"

    def __init__(
        self,
        X: np.ndarray,
        bins: Union[int, str] = "auto",
        alpha: float = 1e-10,
        bound_ext: float = 0.1,
        domain_hint: Optional[List[float]] = None,
        privacy_noise: float = None
    ):
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
        estimators = []

        for i, iX in enumerate(X.T):
            iX_min, iX_max = np.nanmin(iX), np.nanmax(iX)
            diff = iX_max - iX_min
            lower_bound = iX_min - bound_ext * diff
            upper_bound = iX_max + bound_ext * diff
            if isinstance(bins, str) and bins == "auto":
                bins = auto_bins(iX)
            hist = np.histogram(
                iX, bins=bins, range=(lower_bound, upper_bound))

            if privacy_noise is not None:
                noise = np.random.normal(0, privacy_noise, bins)
                noisy_hist = (hist[0].astype(float) +
                              noise).astype(int), hist[1]
                i_estimator = rv_histogram(noisy_hist)
            else:
                i_estimator = rv_histogram(hist)

            i_estimator._hpdf += alpha  # regularization

            estimators.append(i_estimator)

        self.estimators = estimators

    def forward(self, X):
        Z = np.zeros_like(X)
        for idim, iX in enumerate(X.T):
            Z[:, idim] = self.estimators[idim].cdf(iX)

            # When adding a large amount of noise for privacy, the histogram uniformizer may return NaNs after several iterations.
            # If this happens, the NaNs are replaced by random values within the dimension domain.
            if np.isnan(Z[:, idim]).any():
                dim_min = np.nanmin(Z[:, idim])
                dim_max = np.nanmax(Z[:, idim])
                nan_idx = np.isnan(Z[:, idim])
                if np.isnan(dim_min) and np.isnan(dim_max):
                    dim_min = 0.0
                    dim_max = 1.0
                for i_nan_row in np.where(nan_idx)[0]:
                    Z[i_nan_row, idim] = np.random.uniform(dim_min, dim_max)

        return Z

    def inverse(self, Z):
        X = np.zeros_like(Z)
        for idim, iZ in enumerate(Z.T):
            X[:, idim] = self.estimators[idim].ppf(iZ)

        return X

    def gradient(self, X):
        X_grad = np.zeros_like(X)
        for idim, iX in enumerate(X.T):
            X_grad[:, idim] = self.estimators[idim].logpdf(iX)
        X_grad = X_grad.sum(axis=-1)
        return X_grad


class KDEParams(NamedTuple):
    support: np.ndarray
    pdf_est: np.ndarray
    cdf_est: np.ndarray


class MarginalKDEUniformization:
    name: str = "kdefft"

    def __init__(
        self,
        X: np.ndarray,
        grid_size: int = 50,
        n_quantiles: int = 1_000,
        bound_ext: float = 0.1,
        fft: bool = True,
    ):
        """Uniformize marginal distributions using the Kernel Distribution of
        the marginal of another dataset.

        Args:
            X (np.ndarray): dataset
            bins (Union[int, str]): histogram estimation
            alpha (float): regularization on the histogram
            bound_ext (float): tolerance factor on the histogram domain
            domain_hint: Optional[List[float]]: Bounds on the marginal domains
        """
        estimators = []

        # estimate bandwidth
        bw = np.power(X.shape[0], -1 / (X.shape[1] + 4.0))

        for iX in X.T:
            # create histogram
            estimator = sm.nonparametric.KDEUnivariate(iX.squeeze())

            estimator.fit(
                kernel="gau",
                bw=bw,
                fft=fft,
                gridsize=grid_size,
            )

            # estimate support
            diff = iX.max() - iX.min()
            lower_bound = iX.min() - bound_ext * diff
            upper_bound = iX.max() + bound_ext * diff
            support = np.linspace(lower_bound, upper_bound, n_quantiles)

            # estimate empirical pdf from data
            hpdf = estimator.evaluate(support)

            # estimate empirical cdf from data
            hcdf = ECDF(iX)(support)

            kde_params = KDEParams(
                support=support, pdf_est=np.log(hpdf), cdf_est=hcdf)
            estimators.append(kde_params)

        self.estimators = estimators

    def forward(self, X):

        Z = np.zeros_like(X)
        for idim, iX in enumerate(X.T):
            iparams = self.estimators[idim]
            Z[:, idim] = np.interp(iX, xp=iparams.support, fp=iparams.cdf_est)

        return Z

    def inverse(self, Z):
        X = np.zeros_like(Z)
        for idim, iZ in enumerate(Z.T):
            iparams = self.estimators[idim]
            X[:, idim] = np.interp(iZ, xp=iparams.cdf_est, fp=iparams.support)
        return X

    def gradient(self, X):
        X_grad = np.zeros_like(X)
        for idim, iX in enumerate(X.T):
            iparams = self.estimators[idim]
            X_grad[:, idim] = np.interp(
                iX, xp=iparams.support, fp=iparams.pdf_est)
        X_grad = X_grad.sum(axis=-1)
        return X_grad
