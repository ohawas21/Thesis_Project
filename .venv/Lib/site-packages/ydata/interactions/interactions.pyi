from ydata.core.enum import StringEnum
from ydata.dataset import Dataset
from ydata.metadata import Metadata

class InteractionType(StringEnum):
    RECTANGULAR_BINNING: str

class InteractionEngine:
    def __init__(self) -> None: ...
    def calculate(self, dataset: Dataset, metadata: Metadata, interaction_type: InteractionType | str = ..., num_intervals: int = 15) -> list: ...
