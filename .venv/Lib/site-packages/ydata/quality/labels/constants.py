"""File to define all the labels methods defined constants."""

FLOATING_POINT_COMPARISON = 1e-6  # floating point comparison for fuzzy equals
CONFIDENT_THRESHOLDS_LOWER_BOUND = (
    2 * FLOATING_POINT_COMPARISON
)  # lower bound imposed to clip confident thresholds from below, has to be larger than floating point comparison
TINY_VALUE = 1e-100
FLOATING_POINT_COMPARISON = 1e-6  # floating point comparison for fuzzy equals
CLIPPING_LOWER_BOUND = 1e-6
