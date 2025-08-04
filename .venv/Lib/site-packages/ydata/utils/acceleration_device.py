"""ENUM class to define the available and validate acceleration devices."""
from enum import Enum


class Device(Enum):
    CPU = "cpu"
    GPU = "gpu"
