from _typeshed import Incomplete
from pandas import DataFrame as pdDataFrame
from ydata.metadata import Metadata as Metadata
from ydata.metadata.column import Column as Column
from ydata.synthesizers.base_synthesizer import BaseSynthesizer
from ydata.utils.random import RandomSeed as RandomSeed

methods: Incomplete
methods_map: Incomplete

class SeqCartHierarchical(BaseSynthesizer):
    DEVICE: Incomplete
    proper: Incomplete
    default_method: Incomplete
    random_state: Incomplete
    order: Incomplete
    smoothing: Incomplete
    smoothing_strategy: Incomplete
    visit_sequence: Incomplete
    columns_info: Incomplete
    max_n_samples: Incomplete
    col_to_method: Incomplete
    col_to_function: Incomplete
    origin_dates: Incomplete
    def __init__(self, proper: bool = False, smoothing: str = 'NA', default_method: str = 'cart', random_state: RandomSeed = None, regression_order: int = 5) -> None:
        """Sequential Hierarchical Cart initialization.

        Args:
            proper (bool): True is proper synthesization, False otherwise.d
            smoothing (str): Smoothing strategy. No smoothing applied by default.
            default_method (str): Default method for column synthesization.
            random_state (Optional[int]): Random state of the synthesizer.
            regression_order (int): Order for the auto-regressive columns extraction.
        """
    saved_methods: Incomplete
    def fit(self, X: pdDataFrame, metadata: Metadata, dtypes: dict[str, Column] | None = None, extracted_cols: list[str] | None = None, bootstrapping_cols: list[str] | None = None) -> SeqCartHierarchical:
        """Fit the SeqCartHierarchical synthesizer models to the provided training data.

        Args:
            X (Dataset): Training data.
            metadata (Metadata): The meta info from the provided dataset
            extracted_cols (List[str]): List of columns which have been extracted.
            bootstrapping_cols (List[str]): List of columns to boostrap the process

        Returns:
            SeqCartHierarchical: Synthesizer instance
        """
    def sample(self, n_samples: int = 100, bootstrapping: pdDataFrame | None = None, random_state: RandomSeed = None) -> pdDataFrame:
        """Generate a sample of synthetic data.

        Args:
            n_samples (int): Sample size.
            bootstrapping (DataFrame, optional): Data for the bootstrapping columns

        Returns:
            pd.DataFrame: Synthetic data
        """
    def save(self, path: str): ...
    @staticmethod
    def load(path: str) -> SeqCartHierarchical: ...
