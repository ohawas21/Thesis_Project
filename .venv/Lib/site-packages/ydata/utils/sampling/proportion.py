"""Proportion.

The following functions allow us to calculate the estimated proportion of a binary outcome
within a specified margin of error and confidence level using either the continuity corrected
Wilson interval (preferred method) or the Wald interval.

Both the reasoning and equations can be found in the research repository under sampling/sampling_characteristics.py
"""
import math
from typing import Tuple

from numpy import clip
from scipy.stats import norm

# Pre-computed for a threshold of 0.80, indexed on confidence_level
SAMPLE_SIZES = {
    0.99: 11000,  # Exact value: 10538
    0.95: 250  # Exact value: 245
}


def calculate_wilson_sample_size(n: int, p: float, z: float, e: float) -> int:
    """calculate the sample size to achieve the desired margin of error at a
    specified confidence level.

    Args:
        n (int): total population size
        p (float): estimated proportion
        z (float): z score (calculated from the desired confidence level)
        e (float): margin of error

    Returns:
        int: sample size
    """
    numerator = n * z**2 * p * (1 - p)
    denom = e**2 * (n - 1) + z**2 * p * (1 - p)
    sample_pop = numerator / denom
    return int(sample_pop)


def calculate_wilson_cc(
    z: float, n: int, p: float
) -> Tuple[float, float, float, float]:
    """calculate the continuity corrected wilson score interval.

    Args:
        z (float): z score (calculated from the desired confidence level)
        n (int): sample population size
        p (float): unadjusted proportion

    Returns:
        Tuple[float, float, float, float]:
        adjusted_p, margin_of_error, upper_bound, lower_bound
    """
    denominator = 2 * (n + z**2)
    adjusted_moe = (
        z * math.sqrt(z**2 - (1 / n) + 4 * n * p * (1 - p) + (4 * p - 2)) + 1
    ) / denominator
    adjusted_p = clip((2 * n * p + z**2) / denominator, a_min=0, a_max=1)
    upper_bound = min(1, adjusted_p + adjusted_moe)
    lower_bound = max(0, adjusted_p - adjusted_moe)

    margin_of_error = (upper_bound - lower_bound) / 2

    return adjusted_p, margin_of_error, upper_bound, lower_bound


def determine_z_value(confidence_level: float) -> float:
    return norm.ppf(1 - (1 - confidence_level) / 2)


def is_normal(n: int, p: float) -> bool:
    return (n * p) > 5 and (n * (1 - p)) > 5


def calculate_normal_sample_size(
    confidence_level: float = 0.99, margin_error: float = 0.05
) -> int:
    z_value = determine_z_value(confidence_level)
    sample_size = ((z_value * 0.5) / margin_error) ** 2
    sample_size = int(math.ceil(sample_size))

    return sample_size


def calculate_normal_margin_of_error(z_value: float, p: float, sample_size: int):
    return z_value * math.sqrt(p * (1 - p) / sample_size)
