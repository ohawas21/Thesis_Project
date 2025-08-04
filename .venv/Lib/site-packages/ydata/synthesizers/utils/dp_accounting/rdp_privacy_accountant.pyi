from _typeshed import Incomplete
from typing import Sequence

def compute_delta(orders: Sequence[float], rdp: Sequence[float], epsilon: float) -> tuple[float, int]:
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
def compute_epsilon(orders: Sequence[float], rdp: Sequence[float], delta: float) -> tuple[float, int]:
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

class RdpAccountant:
    """Privacy accountant that uses Renyi differential privacy."""
    DEFAULT_RDP_ORDERS: Incomplete
    def __init__(self, orders: Sequence[float] | None = None) -> None: ...
    def compose_gaussian_dp_event(self, noise_multiplier): ...
    def get_epsilon(self, target_delta: float) -> float: ...
    def get_delta(self, target_epsilon: float) -> float: ...
