from _typeshed import Incomplete
from pandas import DataFrame as pdDataFrame
from pathlib import Path
from typing import Callable
from ydata.connectors.storages.big_query_connector import BigQueryConnector as BigQueryConnector
from ydata.connectors.storages.object_storage_connector import ObjectStorageConnector as ObjectStorageConnector
from ydata.connectors.storages.rdbms_connector import RDBMSConnector as RDBMSConnector
from ydata.datascience.common import PrivacyLevel
from ydata.dataset import Dataset
from ydata.metadata.metadata import Metadata as Metadata
from ydata.preprocessors.methods.anonymization import AnonymizerConfigurationBuilder
from ydata.synthesizers.base_model import BaseModel, SegmentByType
from ydata.synthesizers.conditional import ConditionalFeature
from ydata.utils.random import RandomSeed as RandomSeed

logger: Incomplete

class RegularSynthesizer(BaseModel):
    '''
    RegularSynthesizer: A tool for generating high-quality tabular synthetic data.

    The `RegularSynthesizer` is designed to learn patterns from real datasets
    and generate synthetic data that maintains statistical properties while ensuring
    privacy and security. It provides a simple API for training, sampling, saving,
    and loading models.

    ### Key Features
    - **fit **: Learn from real data to create a generative model.
    - **sample **: Produce high-quality synthetic data based on the trained model.
    - **save **: Store a trained synthesizer for future use.
    - **load **: Restore a previously trained synthesizer.

    ### Usage Example
    ```python
    from ydata.synthesizers import RegularSynthesizer

    # Step 1: Train the model
    synth = RegularSynthesizer()
    synth.fit(real_data, metadata)

    # Step 2: Generate synthetic data
    synthetic_data = synth.sample(n_samples=1000)

    # Step 3: Save the trained model
    synth.save("model.pkl")

    # Step 4: Load the trained model later
    loaded_synth = RegularSynthesizer.load("model.pkl")
    ```
    '''
    filter_outliers: Incomplete
    def __init__(self, *, tmppath: str | Path = None, filter_outliers: bool = True, strategy: str = 'random') -> None: ...
    @property
    def SUPPORTED_DTYPES(self): ...
    anonymize: Incomplete
    is_fitted_: bool
    def fit(self, X: Dataset, metadata: Metadata, *, condition_on: str | list[str] | None = None, privacy_level: PrivacyLevel | str = ..., calculated_features: list[dict[str, str | Callable | list[str]]] | None = None, anonymize: dict | AnonymizerConfigurationBuilder | None = None, anonymize_ids: bool = False, segment_by: SegmentByType = 'auto', holdout_size: float = 0.2, random_state: RandomSeed = None):
        '''
        Train the `RegularSynthesizer` on real tabular data.

        This method learns patterns from the provided dataset (`X`) to build a generative
        model capable of producing high-quality synthetic data. It allows for feature
        extraction, handling missing values, and applying privacy controls.

        - **Handles missing values** and applies anonymization if required.
        - **Supports conditional synthesis** by segmenting data into meaningful groups.
        - **Integrates business rules** through the calculated features to evaluate model performance.

        Args:
            X: The real dataset used to train the synthesizer.
            metadata: object describing the dataset, including feature types and relationships.
            calculated_features: List of computed features that should be derived before training, if provided
            anonymize: Configuration for anonymization strategies, such as hashing or generalization, if provided
            privacy_level: Defines the trade-off between privacy and data fidelity.  **Options:** `"HIGH_FIDELITY"`, `"BALANCED_PRIVACY_FIDELITY"`, `"HIGH_PRIVACY"`. Defaults to `"HIGH_FIDELITY"`.
            condition_on: Enables **conditional data generation** by specifying key features to condition the model on.
            anonymize_ids: If `True`, automatically anonymizes columns of type ID. Defaults to `False`.
            segment_by: Defines how data should be segmented while training, based on a column or an automated decision.  **Options:** `"auto"` (default).
            holdout_size: Percentage of data to **hold out** for model evaluation. Default is `0.2` (20%).
            random_state: Set a **seed** for reproducibility. If `None`, randomness is used.

        Returns:
            None: Trains the synthesizer in place.
            None: Trains the synthesizer in place.

        '''
    def sample(self, n_samples: int = 1, condition_on: list[ConditionalFeature] | dict | pdDataFrame | None = None, balancing: bool = False, random_state: RandomSeed = None, connector: BigQueryConnector | ObjectStorageConnector | RDBMSConnector | None = None, **kwargs):
        """
        Generate synthetic tabular data using the trained `RegularSynthesizer`.

        This method generates new synthetic records that mimic the statistical
        properties of the original dataset. Users can optionally **condition**
        on specific features, apply **balancing strategies**, and define an output
        **storage connector** for direct integration with databases or cloud storage.

        Args:
            n_samples: Number of synthetic records/rows to generate. Default is `1`.
            condition_on: **Condition the generator** on specific feature values to create data with controlled distributions.
            balancing: If `True`, ensures balanced sampling the defined conditional features. Default is `False`.
            random_state: Set a **random seed** for reproducibility. Default is `None` (random generation).
            connector:  If provided, the generated synthetic data is automatically **stored** in a cloud-based data warehouse or database.

        Returns:
            sample (Dataset): A Dataset object containing the synthetic samples.
        """
