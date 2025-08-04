from enum import Enum

class LabelFilter(Enum):
    CONFIDENT_LEARNING = 'confident_learning'
    PRUNE = 'prune_by_noise_rate'
