# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Privacy accountant that uses Renyi differential privacy.

Adapted from https://github.com/google/differential-privacy/blob/main/python/dp_accounting/rdp/rdp_privacy_accountant.py.
"""

import math
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from scipy import special


def _log_add(logx: float, logy: float) -> float:
    """Adds two numbers in the log space."""
    a, b = min(logx, logy), max(logx, logy)
    if a == -np.inf:  # adding 0
        return b
    # Use exp(a) + exp(b) = (exp(a - b) + 1) * exp(b)
    return math.log1p(math.exp(a - b)) + b  # log1p(x) = log(x + 1)


def _log_sub(logx: float, logy: float) -> float:
    """Subtracts two numbers in the log space.

    Answer must be non-negative.
    """
    if logx < logy:
        raise ValueError('The result of subtraction must be non-negative.')
    if logy == -np.inf:  # subtracting 0
        return logx
    if logx == logy:
        return -np.inf  # 0 is represented as -np.inf in the log space.

    try:
        # Use exp(x) - exp(y) = (exp(x - y) - 1) * exp(y).
        # expm1(x) = exp(x) - 1
        return math.log(math.expm1(logx - logy)) + logy
    except OverflowError:
        return logx


def _log_comb(n: int, k: int) -> float:
    """Computes log of binomial coefficient."""
    return (special.gammaln(n + 1) - special.gammaln(k + 1) -
            special.gammaln(n - k + 1))


def _compute_log_a_int(q: float, sigma: float, alpha: int) -> float:
    """Computes log(A_alpha) for integer alpha, 0 < q < 1."""

    # Initialize with 0 in the log space.
    log_a = -np.inf

    for i in range(alpha + 1):
        log_coef_i = (
            _log_comb(alpha, i) + i * math.log(q) + (alpha - i) * math.log(1 - q))

        s = log_coef_i + (i * i - i) / (2 * (sigma**2))
        log_a = _log_add(log_a, s)

    return float(log_a)


def _compute_log_a_frac(q: float, sigma: float, alpha: float) -> float:
    """Computes log(A_alpha) for fractional alpha, 0 < q < 1."""
    # The two parts of A_alpha, integrals over (-inf,z0] and [z0, +inf), are
    # initialized to 0 in the log space:
    log_a0, log_a1 = -np.inf, -np.inf
    i = 0

    z0 = sigma**2 * math.log(1 / q - 1) + .5

    while True:  # do ... until loop
        coef = special.binom(alpha, i)
        log_coef = math.log(abs(coef))
        j = alpha - i

        log_t0 = log_coef + i * math.log(q) + j * math.log(1 - q)
        log_t1 = log_coef + j * math.log(q) + i * math.log(1 - q)

        log_e0 = math.log(.5) + _log_erfc((i - z0) / (math.sqrt(2) * sigma))
        log_e1 = math.log(.5) + _log_erfc((z0 - j) / (math.sqrt(2) * sigma))

        log_s0 = log_t0 + (i * i - i) / (2 * (sigma**2)) + log_e0
        log_s1 = log_t1 + (j * j - j) / (2 * (sigma**2)) + log_e1

        if coef > 0:
            log_a0 = _log_add(log_a0, log_s0)
            log_a1 = _log_add(log_a1, log_s1)
        else:
            log_a0 = _log_sub(log_a0, log_s0)
            log_a1 = _log_sub(log_a1, log_s1)

        i += 1
        if max(log_s0, log_s1) < -30:
            break

    return _log_add(log_a0, log_a1)


def _log_erfc(x: float) -> float:
    """Computes log(erfc(x)) with high accuracy for large x."""
    try:
        return math.log(2) + special.log_ndtr(-x * 2**.5)
    except NameError:
        # If log_ndtr is not available, approximate as follows:
        r = special.erfc(x)
        if r == 0.0:
            # Using the Laurent series at infinity for the tail of the erfc function:
            #     erfc(x) ~ exp(-x^2-.5/x^2+.625/x^4)/(x*pi^.5)
            # To verify in Mathematica:
            #     Series[Log[Erfc[x]] + Log[x] + Log[Pi]/2 + x^2, {x, Infinity, 6}]
            return (-math.log(math.pi) / 2 - math.log(x) - x**2 - .5 * x**-2 +
                    .625 * x**-4 - 37. / 24. * x**-6 + 353. / 64. * x**-8)
        else:
            return math.log(r)


def compute_delta(orders: Sequence[float], rdp: Sequence[float],
                  epsilon: float) -> Tuple[float, int]:
    """Computes delta given a list of RDP values and target epsilon.

    Args:
        orders: An array of orders.
        rdp: An array of RDP guarantees.
        epsilon: The target epsilon.

    Returns:
        2-tuple containing optimal delta and the optimal order.

    Raises:
        ValueError: If input is malformed.
    """
    if epsilon < 0:
        raise ValueError(f'Epsilon cannot be negative. Found {epsilon}.')
    if len(orders) != len(rdp):
        raise ValueError('Input lists must have the same length.')

    # Basic bound (see https://arxiv.org/abs/1702.07476 Proposition 3 in v3):
    #   delta = min( np.exp((rdp - epsilon) * (orders - 1)) )

    # Improved bound from https://arxiv.org/abs/2004.00010 Proposition 12 (in v4):
    logdeltas = []  # work in log space to avoid overflows
    for (a, r) in zip(orders, rdp):
        if a < 1:
            raise ValueError(
                f'Renyi divergence order must be at least 1. Found {a}.')
        if r < 0:
            raise ValueError(
                f'Renyi divergence cannot be negative. Found {r}.')
        # For small alpha, we are better of with bound via KL divergence:
        # delta <= sqrt(1-exp(-KL)).
        # Take a min of the two bounds.
        if r == 0:
            logdelta = -np.inf
        else:
            logdelta = 0.5 * math.log1p(-math.exp(-r))
        if a > 1.01:
            # This bound is not numerically stable as alpha->1.
            # Thus we have a min value for alpha.
            # The bound is also not useful for small alpha, so doesn't matter.
            rdp_bound = (a - 1) * (r - epsilon +
                                   math.log1p(-1 / a)) - math.log(a)
            logdelta = min(logdelta, rdp_bound)

        logdeltas.append(logdelta)

    optimal_index = np.argmin(logdeltas)
    return min(math.exp(logdeltas[optimal_index]), 1.), orders[optimal_index]


def compute_epsilon(orders: Sequence[float], rdp: Sequence[float],
                    delta: float) -> Tuple[float, int]:
    """Computes epsilon given a list of RDP values and target delta.

    Args:
        orders: An array of orders.
        rdp: An array of RDP guarantees.
        delta: The target delta. Must be >= 0.

    Returns:
        2-tuple containing optimal epsilon and the optimal order.

    Raises:
        ValueError: If input is malformed.
    """
    if delta < 0:
        raise ValueError(f'Delta cannot be negative. Found {delta}.')

    if delta == 0:
        if all(r == 0 for r in rdp):
            return 0, 0
        else:
            return np.inf, 0

    if len(orders) != len(rdp):
        raise ValueError('Input lists must have the same length.')

    # Basic bound (see https://arxiv.org/abs/1702.07476 Proposition 3 in v3):
    #   epsilon = min( rdp - math.log(delta) / (orders - 1) )

    # Improved bound from https://arxiv.org/abs/2004.00010 Proposition 12 (in v4).
    # Also appears in https://arxiv.org/abs/2001.05990 Equation 20 (in v1).
    eps = []
    for (a, r) in zip(orders, rdp):
        if a < 1:
            raise ValueError(
                f'Renyi divergence order must be at least 1. Found {a}.')
        if r < 0:
            raise ValueError(
                f'Renyi divergence cannot be negative. Found {r}.')

        if delta**2 + math.expm1(-r) > 0:
            # In this case, we can simply bound via KL divergence:
            # delta <= sqrt(1-exp(-KL)).
            # No need to try further computation if we have epsilon = 0.
            epsilon = 0
        elif a > 1.01:
            # This bound is not numerically stable as alpha->1.
            # Thus we have a min value of alpha.
            # The bound is also not useful for small alpha, so doesn't matter.
            epsilon = r + math.log1p(-1 / a) - math.log(delta * a) / (a - 1)
        else:
            # In this case we can't do anything. E.g., asking for delta = 0.
            epsilon = np.inf
        eps.append(epsilon)

    optimal_index = np.argmin(eps)
    return max(0, eps[optimal_index]), orders[optimal_index]


def _compute_log_a(q: float, noise_multiplier: float,
                   alpha: Union[int, float]) -> float:
    if float(alpha).is_integer():
        return _compute_log_a_int(q, noise_multiplier, int(alpha))
    else:
        return _compute_log_a_frac(q, noise_multiplier, alpha)


def _compute_rdp_poisson_subsampled_gaussian(
        q: float, noise_multiplier: float,
        orders: Sequence[float]) -> Union[float, np.ndarray]:
    """Computes RDP of the Poisson sampled Gaussian mechanism.

    Args:
        q: The sampling rate.
        noise_multiplier: The ratio of the standard deviation of the Gaussian noise
        to the l2-sensitivity of the function to which it is added.
        orders: An array of RDP orders.

    Returns:
        The RDPs at all orders. Can be `np.inf`.
    """

    def compute_one_order(q, alpha):
        if q == 0:
            return 0

        if np.isinf(alpha) or noise_multiplier == 0:
            return np.inf

        if q == 1.:
            return alpha / (2 * noise_multiplier**2)

        return _compute_log_a(q, noise_multiplier, alpha) / (alpha - 1)

    return np.array([compute_one_order(q, order) for order in orders])


class RdpAccountant():
    """Privacy accountant that uses Renyi differential privacy."""

    # Default orders chosen to give good coverage for Gaussian mechanism in
    # the privacy regime of interest.
    DEFAULT_RDP_ORDERS = ([1 + x / 10. for x in range(1, 100)] +
                          list(range(11, 64)) + [128, 256, 512, 1024])

    def __init__(
        self,
        orders: Optional[Sequence[float]] = None
    ):
        if orders is None:
            orders = self.DEFAULT_RDP_ORDERS
        self._orders = np.array(orders)
        self._rdp = np.zeros_like(orders, dtype=np.float64)

    def compose_gaussian_dp_event(self, noise_multiplier):
        self._rdp += _compute_rdp_poisson_subsampled_gaussian(
            q=1.0, noise_multiplier=noise_multiplier, orders=self._orders)
        return self

    def get_epsilon(self, target_delta: float) -> float:
        return compute_epsilon(self._orders, self._rdp, target_delta)[0]

    def get_delta(self, target_epsilon: float) -> float:
        return compute_delta(self._orders, self._rdp, target_epsilon)[0]
