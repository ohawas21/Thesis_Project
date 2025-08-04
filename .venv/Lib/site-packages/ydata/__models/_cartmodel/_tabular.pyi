from _typeshed import Incomplete
from pandas import DataFrame
from ydata.__models._cartmodel.maps import CART_FLAVOR, METHODS_MAP
from ydata.metadata import Metadata as Metadata
from ydata.metadata.column import Column as Column
from ydata.synthesizers.base_synthesizer import BaseSynthesizer
from ydata.utils.random import RandomSeed as RandomSeed

methods: Incomplete
methods_map: Incomplete

class CartHierarchical(BaseSynthesizer):
    FLAVOR: Incomplete
    DEVICE: Incomplete
    visit_sequence: Incomplete
    predictor_matrix: Incomplete
    columns_info: Incomplete
    proper: Incomplete
    smoothing: Incomplete
    smoothing_strategy: Incomplete
    default_method: Incomplete
    col_to_method: Incomplete
    col_to_function: Incomplete
    random_state: Incomplete
    def __init__(self, proper: bool = False, smoothing: str = 'NA', default_method: str = 'cart', random_state: RandomSeed = None) -> None:
        """Tabular cart initialization.

        Args:
            proper (bool): True is proper synthesization, False otherwise.d
            smoothing (str): Smoothing strategy. No smoothing applied by default.
            default_method (str): Default method for column synthesization.
            random_state: Random state of the synthesizer.
        """
    saved_methods: Incomplete
    def fit(self, X: DataFrame, metadata: Metadata, dtypes: dict[str, Column] = None, method: list | METHODS_MAP[CART_FLAVOR.TAB] | None = None, cont_na: dict | None = None, bootstrapping_cols: list[str] | None = None) -> CartHierarchical:
        """Fit the cart synthesizer models to the provided training data.

        Args:
            X (Dataset): Training data.
            metadata (Metadata): The meta info from the provided dataset.
            method (List[str]): List of methods to apply on each columns.
            cont_na: (Dict): Dictionary indicating the missing values replacement for continuous variables.
            bootstrapping_cols (List[str]): List of columns to boostrap the process (not used for regular at the moment)

        Returns:
            SeqCart: Synthesizer instance
        """
    def sample(self, n_samples: int = 100, bootstrapping: DataFrame | None = None, random_state: RandomSeed = None) -> DataFrame:
        """Generate a sample of synthetic data.

        Args:
            n_samples (int): Sample size.
            bootstrapping (DataFrame, optional): Data for the bootstrapping columns

        Returns:
            pd.DataFrame: Synthetic data
        """
    def save(self, path: str): ...
    @staticmethod
    def load(path: str) -> CartHierarchical: ...
