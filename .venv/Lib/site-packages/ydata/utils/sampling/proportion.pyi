from _typeshed import Incomplete

SAMPLE_SIZES: Incomplete

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
def calculate_wilson_cc(z: float, n: int, p: float) -> tuple[float, float, float, float]:
    """calculate the continuity corrected wilson score interval.

    Args:
        z (float): z score (calculated from the desired confidence level)
        n (int): sample population size
        p (float): unadjusted proportion

    Returns:
        Tuple[float, float, float, float]:
        adjusted_p, margin_of_error, upper_bound, lower_bound
    """
def determine_z_value(confidence_level: float) -> float: ...
def is_normal(n: int, p: float) -> bool: ...
def calculate_normal_sample_size(confidence_level: float = 0.99, margin_error: float = 0.05) -> int: ...
def calculate_normal_margin_of_error(z_value: float, p: float, sample_size: int): ...
