from typing import Sequence

class BracketInterval: ...

class ExplicitBracketInterval(BracketInterval):
    endpoint_1: float
    endpoint_2: float

class LowerEndpointAndGuess(BracketInterval):
    lower_endpoint: float
    initial_guess: float

class NoBracketIntervalFoundError(Exception):
    """Error raised when explicit bracket interval cannot be found."""
class NoOptimumFoundError(Exception):
    """Error raised when root finding algorithm fails."""

def calibrate_dp_mechanism(target_epsilon: float, target_delta: float, bracket_interval: BracketInterval | None = None, orders: Sequence[float] | None = None, discrete: bool = False, tol: float | None = None) -> float | int:
    """Searches for optimal mechanism parameter value within privacy budget.

    The procedure searches over the space of parameters by creating, for each
    sample value, a Gaussian event representing the mechanism generated from
    that value, and a freshly initialized PrivacyAccountant. Then the accountant
    is applied to the event to determine its epsilon at the target delta. Brent's
    method is usedto determine the value of the parameter at which the target
    epsilon is achieved.

    Args:
        target_epsilon: The target epsilon value.
        target_delta: The target delta value.
        bracket_interval: A BracketInterval used to determine the upper and lower
        endpoints of the interval within which Brent's method will search. If
        None, searches for a non-negative bracket starting from [0, 1].
        discrete: A bool determining whether the parameter is continuous or discrete
        valued. If True, the parameter is assumed to take only integer values.
        Concretely, `discrete=True` has three effects. 1) ints, not floats are
        passed to `make_event_from_param`. 2) The minimum optimization tolerance
        is 0.5. 3) An integer is returned.
        tol: The tolerance, in parameter space. If the maximum (or minimum) value of
        the parameter that meets the privacy requirements is x*,
        calibrate_dp_mechanism is guaranteed to return a value x such that |x -
        x*| <= tol. If `None`, tol is set to 1e-6 for continuous parameters or 0.5
        for discrete parameters.

    Returns:
        A value of the parameter within tol of the optimum subject to the privacy
        constraint. If discrete=True, the returned value will be an integer.
        Otherwise it will be a float.

    Raises:
        NoBracketIntervalFoundError: if bracket_interval is LowerEndpointAndGuess
        and no upper bound can be found within a factor of 2**30 of the original
        guess.
        NoOptimumFoundError: if scipy.optimize.brentq fails to find an optimum.
    """
