from _typeshed import Incomplete
from pathlib import Path
from ydata.connectors.storages.big_query_connector import BigQueryConnector as BigQueryConnector
from ydata.dataset import Dataset
from ydata.metadata.column import Column as Column
from ydata.synthesizers.exceptions import SynthesizerValueError as SynthesizerValueError
from ydata.utils.exceptions import NotEnoughRows as NotEnoughRows
from ydata.utils.random import RandomSeed as RandomSeed

logger: Incomplete
metrics_logger: Incomplete
SegmentByType: Incomplete

def get_default_tmppath(folder: Path = None) -> Path: ...

class BaseModel:
    """Base class for the synthesis models."""
    is_fitted_: bool
    DEFAULT_LAB_TMP: str
    MIN_ROWS: int
    LOW_ROWS: int
    calculated_features: Incomplete
    pipelines: Incomplete
    segment_by: str
    dataset_preprocessor: Incomplete
    features_order: Incomplete
    data_types: Incomplete
    fitted_dataset_schema: Incomplete
    segmenter: Incomplete
    slicer: Incomplete
    segmentation_strategy: Incomplete
    slicing_strategy: Incomplete
    dataset_type: Incomplete
    uuid: Incomplete
    tmppath: Incomplete
    anonymize: Incomplete
    entity_augmenter: Incomplete
    pivot_columns: Incomplete
    entities_type: Incomplete
    time_series: bool
    metadata_summary: Incomplete
    categorical_vars: Incomplete
    random_state: Incomplete
    def __init__(self, tmppath: str | Path = None) -> None: ...
    @property
    def privacy_level(self): ...
    @property
    def anonymized_columns(self) -> list[str]: ...
    def fit(self, X, y: Incomplete | None = None, segment_by: SegmentByType = 'auto', anonymize: dict | None = None, random_state: RandomSeed = None): ...
    def sample(self, n_samples: int | None = None, n_entities: int | None = None) -> Dataset: ...
    @property
    def SUPPORTED_DTYPES(self) -> None: ...
    def save(self, path: str, copy: bool = False):
        """
        Save the trained `RegularSynthesizer` model to disk.

        This method serializes and stores the trained synthesizer, allowing users
        to reload and reuse it later without retraining. The saved model includes
        all learned patterns, privacy constraints, and configuration settings.

        Args:
            path: The file path where the trained model should be saved. The file should have a `.pkl` or similar extension for serialization.
            copy: If `True`, a deep copy of the synthesizer is saved instead of the original. Defaults to `False`.
        """
    @staticmethod
    def load(path):
        '''Load a previously trained `RegularSynthesizer` from disk.

        This method restores a synthesizer that was previously trained and saved using
        the `save()` method. It allows users to resume synthetic data generation without retraining.

        Args:
            path str:
                 The file path where the trained synthesizer was saved. Must match the format used in `save()`, typically a `.pkl` file.

        ## Example:
        ```python
        from ydata.synthesizers import RegularSynthesizer

        # Load a previously saved synthesizer
        synth = RegularSynthesizer.load("synthesizer.pkl")

        # Generate synthetic data from the loaded model
        synthetic_data = synth.sample(n_samples=1000)
        ```

        '''
