"""File to define the class enums (This might go to ydata core instead)"""
from enum import Enum


class LabelFilter(Enum):
    CONFIDENT_LEARNING = 'confident_learning'
    PRUNE = 'prune_by_noise_rate'
