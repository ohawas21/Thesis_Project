from _typeshed import Incomplete
from pandas import DataFrame as pdDataFrame
from pathlib import Path
from typing import Callable
from ydata.connectors.storages.big_query_connector import BigQueryConnector as BigQueryConnector
from ydata.connectors.storages.object_storage_connector import ObjectStorageConnector as ObjectStorageConnector
from ydata.connectors.storages.rdbms_connector import RDBMSConnector as RDBMSConnector
from ydata.datascience.common import PrivacyLevel
from ydata.dataset import Dataset
from ydata.metadata import Metadata
from ydata.preprocessors.methods.anonymization import AnonymizerConfigurationBuilder
from ydata.synthesizers.base_model import BaseModel, SegmentByType
from ydata.synthesizers.conditional import ConditionalFeature
from ydata.synthesizers.entity_augmenter import FidelityConfig, SmoothingConfig
from ydata.utils.random import RandomSeed as RandomSeed

logger: Incomplete

class TimeSeriesSynthesizer(BaseModel):
    '''
    TimeSeriesSynthesizer: A tool for generating high-fidelity synthetic time-series data.

    Unlike the `RegularSynthesizer`, the `TimeSeriesSynthesizer` is designed to capture
    and replicate **temporal relationships** within entities over time. It learns from
    sequential patterns in the data and generates synthetic time-series records that
    preserve trends, seasonality, and correlations per entity.

    Additionally, this synthesizer can **augment** datasets by increasing the number of
    unique entities while maintaining realistic temporal behavior.

    ### Key Features
    - **Time-Aware Training (`fit`)**: Learns entity-level sequential dependencies and trends over time.
    - **Pattern-Preserving Sampling (`sample`)**: Generates synthetic time-series data that mimics real-world time progression.
    - **Entity Augmentation**: Expands the dataset by generating additional synthetic entities with realistic time patterns.
    - **Time Window Processing**: Operates on an `N`-entity time window to model time dependencies effectively.
    - **Model Persistence (`save` & `load`)**: Store and restore trained synthesizers for future use.

    To define a single entity series the following Metadata configuration would be required:
    ```python

        dataset_attrs = {
            "sortbykey": "sate",
        }

    metadata = Metadata(dataset, dataset_type=DatasetType.TIMESERIES, dataset_attrs=dataset_attrs)
    ```
    As for a multi-entity time series, it requires the metadata dataset attributes to specify at least one column corresponding to an entity ID.
    For instance, the following example specify two columns as entity ID columns:
    ```python

    dataset_attrs = {
        "sortbykey": "sate",
        "entities": [\'entity\', \'entity_2\']
    }

    metadata = Metadata(dataset, dataset_type=DatasetType.TIMESERIES, dataset_attrs=dataset_attrs)
    ```

    ### Usage Example
    ```python
    from ydata.synthesizers import TimeSeriesSynthesizer

    # Step 1: Train the model with time-series data
    synth = TimeSeriesSynthesizer()
    synth.fit(data, metadata)

    # Step 2: Generate synthetic time-series data
    synthetic_data = synth.sample(n_entities=10)

    # Step 3: Save the trained model
    synth.save("timeseries_model.pkl")

    # Step 4: Load the trained model later
    loaded_synth = TimeSeriesSynthesizer.load("timeseries_model.pkl")
    ```
    '''
    sortbykey: Incomplete
    bypass_entities_anonymization: Incomplete
    def __init__(self, tmppath: str | Path = None) -> None: ...
    @property
    def SUPPORTED_DTYPES(self): ...
    anonymize: Incomplete
    n_entities: Incomplete
    entity_merged_col: Incomplete
    entities_type: Incomplete
    entities_nrows: Incomplete
    is_fitted_: bool
    def fit(self, X: Dataset, metadata: Metadata, extracted_cols: list = None, calculated_features: list[dict[str, str | Callable | list[str]]] | None = None, anonymize: dict | AnonymizerConfigurationBuilder | None = None, privacy_level: PrivacyLevel | str = ..., condition_on: str | list[str] | None = None, anonymize_ids: bool = False, segment_by: SegmentByType = 'auto', random_state: RandomSeed = None):
        '''
        Train the `TimeSeriesSynthesizer` on real time-series data.

        This method learns patterns, dependencies, and sequential behaviors from the input dataset (`X`)
        while preserving the relationships between entities over time. The synthesizer processes **time-dependent**
        features and constructs a generative model capable of producing realistic time-series data.

        Args:
            X (Dataset): Input dataset.
            metadata (Metadata): Metadata instance.
            extracted_cols (list[str]): List of columns to extract data from.
            calculated_features (list[dict[str, str |]]): Defines additional business rules to be ensured for the synthetic generated dataset.
            anonymize (Optional[dict | AnonymizerConfigurationBuilder]): Specifies anonymization strategies for sensitive fields while leveraging ydata\'s AnonymizerEngine
            privacy_level (str | PrivacyLevel): Defines the trade-off between privacy and data fidelity.  **Options:** `"HIGH_FIDELITY"`, `"BALANCED_PRIVACY_FIDELITY"`, `"HIGH_PRIVACY"`. Defaults to `"HIGH_FIDELITY"`. Defaults to `HIGH_FIDELITY`.
            condition_on (Union[str, list[str]]): Enables **conditional data generation** by specifying key features to condition the model on.
            anonymize_ids (bool): If `True`, automatically anonymizes columns of type ID. Defaults to `False`.
            segment_by (str | list | `auto`): Defines how data should be segmented while training, based on a column or an automated decision.  **Options:** `"auto"` (default).
            random_state (Optional): Set a **seed** for reproducibility. If `None`, randomness is used.
        '''
    def sample(self, n_entities: int | None = None, smoothing: bool | dict | SmoothingConfig = False, fidelity: float | dict | FidelityConfig | None = None, sort_result: bool = True, condition_on: list[ConditionalFeature] | dict | pdDataFrame | None = None, balancing: bool = False, random_state: RandomSeed = None, connector: BigQueryConnector | ObjectStorageConnector | RDBMSConnector | None = None, **kwargs) -> Dataset:
        """Generate a time series.

        This method generates a new time series. The instance should be trained via the method `fit` before calling `sample`.
        The generated time series has the same length of the training data. However, in the case of multi-entity time series, it is possible
        The generated time series has the same length of the training data. However, in the case of multi-entity time series, it is possible
        to augment the number of entities by specifying the parameter `n_entities`.

        For a multi-entity sample, there are two major arguments that can be used to modify the results: fidelity and smoothing.

        1. Fidelity: It defines how close the new entities should be from the original ones.
                     When a `float`, it represents the behavioral noise to be added to the entity expressed as a percentage of its variance.
                     See `ydata.synthesizer.entity_augmenter.FidelityConfig` for more details.
        2. Smoothing: It defines if and how the new entities trajectory should be smoothed.
                    See `ydata.synthesizer.entity_augmenter.SmoothingConfig` for more details.

        Args:
            n_entities (Optional[int]): Number of entities to sample. If None, generates as many entities as in the training data. By default None.
            smoothing (Union[bool, dict, SmoothingConfig]): Define how the smoothing should be done. `True` uses the `auto` configuration.
            fidelity Optional[Union[float, dict, FidelityConfig]]: Define the fidely policy.
            sort_result (bool): True if the sample should be sorted by sortbykey, False otherwise.
            condition_on (list[ConditionalFeature] | dict | pdDataFrame | None): Conditional rules to be applied.
            balancing (bool): If True, the categorical features included in the conditional rules have equally distributed percentages.

        Returns:
            Dataset: The generated synthetic time-series dataset
        """
