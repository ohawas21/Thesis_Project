from enum import Enum

class Activation(Enum):
    """Enum for activation functions for the data columns."""
    SOFTMAX = 'softmax'
    TANH = 'tanh'
