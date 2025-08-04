from collections import OrderedDict
from functools import partial
from os import getenv
from pathlib import Path
from typing import Callable, List, Optional, Union
from warnings import warn

from math import ceil

from pandas import DataFrame as pdDataFrame
from pandas import concat, to_numeric

from ydata.connectors.storages.big_query_connector import BigQueryConnector
from ydata.connectors.storages.object_storage_connector import ObjectStorageConnector
from ydata.connectors.storages.rdbms_connector import RDBMSConnector
from ydata.datascience.common import PrivacyLevel
from ydata.dataset import Dataset
from ydata.metadata import Metadata
from ydata.metadata.metadata import DataType
from ydata.preprocessors.identity import Identity as IdentityPreprocessor
from ydata.preprocessors.methods.anonymization import AnonymizerConfigurationBuilder
from ydata.preprocessors.regular.preprocessor import CartHierarchicalPreprocessor, CartHierarchicalSegmentPreprocessor

from ydata.synthesizers.exceptions import SynthesizerValueError
from ydata.synthesizers.logger import synthlogger_config

from ydata.synthesizers.base_model import BaseModel, SegmentByType
from ydata.synthesizers.conditional import ConditionalFeature
from ydata.synthesizers.entity_augmenter import FidelityConfig, SmoothingConfig
from ydata.synthesizers.prototype import PipelinePrototype
from ydata.synthesizers.utils.models import TimeSeriesSynthesizerModel
from ydata.utils.data_types import VariableType
from ydata.utils.exceptions import IgnoredParameter
from ydata.utils.random import RandomSeed
from ydata.utils.misc import log_time_factory

from ydata._licensing import synthesizer_sample

logger = synthlogger_config(verbose=getenv(
    "VERBOSE", "false").lower() == "true")


class TimeSeriesSynthesizer(BaseModel):
    """
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
        "entities": ['entity', 'entity_2']
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
    """
    __name__ = "TimeSeriesSynthesizer"
    _SUPPORTED_DTYPES = [DataType.NUMERICAL, DataType.STR,
                         DataType.CATEGORICAL, DataType.DATE]

    def __init__(self, tmppath: Union[str, Path] = None):
        #Initializes the SYNTHESIZER
        super().__init__(tmppath=tmppath)
        logger.info("[SYNTHESIZER] - Initializing Time Series SYNTHESIZER.")
        self.sortbykey = None
        self.bypass_entities_anonymization = []

    @property
    def SUPPORTED_DTYPES(self):  # noqa: N802
        return self._SUPPORTED_DTYPES

    @staticmethod
    def _validate_timeseries_inputs(dataset: Dataset,
                                    metadata: Metadata) -> None:
        """
            Method to validate time-series specific synthetic data inputs
        """
        attr = metadata.dataset_attrs
        assert attr is not None, "Missing 'dataset_attrs' on Metadata instance"

        if metadata.dataset_attrs is None:
            raise SynthesizerValueError("Missing valid 'dataset_attrs' on Metadata. *'sortbykey'* is a mandatory input.")

        if any([col in [w.column for w in metadata.warnings["constant"]] for col in metadata.dataset_attrs.sortbykey]):
            raise SynthesizerValueError("The *'sortbykey'* defined column cannot be constant. "
                                        "Please validate your dataset profiling and select a new column that is not constant.")


    def _init_metadata(self, X: Dataset, metadata: Metadata):
        if metadata is None:
            logger.info("[SYNTHESIZER] - Calculating metadata")
            metadata = Metadata()
            metadata(X)

        assert isinstance(
            metadata, Metadata
        ), "Please provide a valid Metadata instance."

        return metadata

    def _select_pipeline_prototype(
        self, metadata: Metadata, data_types: Optional[dict] = None
    ) -> PipelinePrototype:
        preprocessor = CartHierarchicalSegmentPreprocessor
        if metadata.shape[0] < 5_000 or metadata.shape[1] < 50 or metadata._is_multitable:
            preprocessor = IdentityPreprocessor
        pipeline_def = PipelinePrototype(
            dataset=PipelinePrototype.DatasetScope(
                preprocessor=CartHierarchicalPreprocessor,
                preprocessor_params={
                    "sortbykey": metadata.dataset_attrs.sortbykey,
                    "anonymize_config": self.anonymize,
                    "metadata": metadata
                },
            ),
            segment=PipelinePrototype.SegmentScope(preprocessor=preprocessor,
                                                   preprocessor_params={
                                                       "metadata": metadata}
                                                   ),
            block=PipelinePrototype.BlockScope(
                synthesizer=TimeSeriesSynthesizerModel.CART.value
            ),
        )
        return pipeline_def

    def fit(
        self,
        X: Dataset,
        metadata: Metadata,
        extracted_cols: List = None,
        calculated_features: list[dict[str, str |
                                       Callable | list[str]]] | None = None,
        anonymize: dict | AnonymizerConfigurationBuilder | None = None,
        privacy_level: PrivacyLevel | str = PrivacyLevel.HIGH_FIDELITY,
        condition_on: Union[str, list[str]] | None = None,
        anonymize_ids: bool = False,
        segment_by: SegmentByType = "auto",
        random_state: RandomSeed = None
    ):
        """
        Train the `TimeSeriesSynthesizer` on real time-series data.

        This method learns patterns, dependencies, and sequential behaviors from the input dataset (`X`)
        while preserving the relationships between entities over time. The synthesizer processes **time-dependent**
        features and constructs a generative model capable of producing realistic time-series data.

        Args:
            X (Dataset): Input dataset.
            metadata (Metadata): Metadata instance.
            extracted_cols (list[str]): List of columns to extract data from.
            calculated_features (list[dict[str, str |]]): Defines additional business rules to be ensured for the synthetic generated dataset.
            anonymize (Optional[dict | AnonymizerConfigurationBuilder]): Specifies anonymization strategies for sensitive fields while leveraging ydata's AnonymizerEngine
            privacy_level (str | PrivacyLevel): Defines the trade-off between privacy and data fidelity.  **Options:** `"HIGH_FIDELITY"`, `"BALANCED_PRIVACY_FIDELITY"`, `"HIGH_PRIVACY"`. Defaults to `"HIGH_FIDELITY"`. Defaults to `HIGH_FIDELITY`.
            condition_on (Union[str, list[str]]): Enables **conditional data generation** by specifying key features to condition the model on.
            anonymize_ids (bool): If `True`, automatically anonymizes columns of type ID. Defaults to `False`.
            segment_by (str | list | `auto`): Defines how data should be segmented while training, based on a column or an automated decision.  **Options:** `"auto"` (default).
            random_state (Optional): Set a **seed** for reproducibility. If `None`, randomness is used.
        """
        if condition_on is not None and isinstance(condition_on, str):
            condition_on = [condition_on]

        privacy_level = PrivacyLevel(privacy_level)

        self._validate_input_values(
            dataset=X,
            metadata=metadata,
            calculated_features=calculated_features,
            condition_on=condition_on,
            segment_by=segment_by,
        )

        self._validate_timeseries_inputs(dataset=X,
                                         metadata=metadata)

        self.anonymize = anonymize
        logger.debug(
            "[SYNTHESIZER] - Updating the metadata to only keep the columns to be synthesized.")

        self._init_fit(metadata, segment_by, calculated_features,
                       anonymize, condition_on, anonymize_ids=anonymize_ids)

        self.sortbykey = metadata.dataset_attrs.sortbykey
        logger.debug("[SYNTHESIZER] - Sorting the dataset values.")
        X = X.copy()

        # Select only the columns to be synthesized
        logger.debug(
            "[SYNTHESIZER] - Selecting only the columns to be synthesized.")
        cols = list(self.data_types.keys())
        X = X[cols]
        metadata = metadata[cols]

        if len(metadata.dataset_attrs.entities) > 0:
            # Adjust anonymizer
            for c in metadata.dataset_attrs.entities:
                if c in self.anonymize.config:
                    self.bypass_entities_anonymization.append(c)
                    self.anonymize.remove_config(c)

            if len(metadata.dataset_attrs.entities) == 1:
                e = metadata.dataset_attrs.entities[0]
                self.n_entities = int(
                    metadata.summary["cardinality"][e])
                self.entities_type[e] = metadata.columns[e].vartype
            else:
                self.entity_merged_col = '|'.join(
                    metadata.dataset_attrs.entities)
                X._data[self.entity_merged_col] = X._data[metadata.dataset_attrs.entities[0]].astype(
                    'string')
                for i in range(1, len(metadata.dataset_attrs.entities)):
                    X._data[self.entity_merged_col] += '|' + \
                        X._data[metadata.dataset_attrs.entities[i]].astype(
                            'string')
                self.entities_type = {
                    k: metadata.columns[k].vartype for k in metadata.dataset_attrs.entities}
                self.n_entities = X._data[self.entity_merged_col].nunique().compute()
        else:
            self.n_entities = 1

        self.entities_nrows = ceil(self.nrows_training / self.n_entities)

        privacy_level = PrivacyLevel(privacy_level)

        logger.debug("[SYNTHESIZER] - Starting the training procedure.")
        self._fit(X, metadata,
                  n_entities=self.n_entities,
                  privacy_level=privacy_level,
                  condition_on=condition_on,
                  fit_params={'extracted_cols': extracted_cols},
                  random_state=random_state)
        self.is_fitted_ = True
        return self

    def _MultiTableSynthesizer__sample(
        self,
        n_entities: int | None = None,
        smoothing: bool | dict | SmoothingConfig = False,
        fidelity: float | dict | FidelityConfig | None = None,
        sort_result: bool = True,
        condition_on: list[ConditionalFeature] | dict | pdDataFrame | None = None,
        balancing: bool = False,
        random_state: RandomSeed = None,
        connector: BigQueryConnector | ObjectStorageConnector | RDBMSConnector | None = None,
        **kwargs
    ) -> Dataset:
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
        self._check_is_fitted()

        _entity_col = self.entities[0] if len(
            self.entities) > 0 else 'entity'

        if len(self.entities) == 0 and (smoothing or fidelity is not None):
            warn("Parameters smoothing and fidelity are ignored when no entity columns is defined in the dataset attributes.", IgnoredParameter)

        if self.entity_augmenter is None:
            if n_entities is None:
                n_entities = 1
            entities = []
            for i in range(n_entities):
                entity = super()._sample(n_samples=self.nrows_training,
                                         smoothing=smoothing, fidelity=fidelity,
                                         condition_on=condition_on, balancing=balancing, random_state=random_state, bypass_entities_anonymization=self.bypass_entities_anonymization)
                if n_entities > 1:
                    entity[_entity_col] = f"{_entity_col}_{i}"
                entities.append(entity)
            result = concat(entities)
        else:
            result = super()._sample(n_samples=n_entities, smoothing=smoothing,
                                     fidelity=fidelity, condition_on=condition_on,
                                     balancing=balancing, random_state=random_state, bypass_entities_anonymization=self.bypass_entities_anonymization)
        if sort_result:
            result.sort_values(by=self.sortbykey,
                               inplace=True, ignore_index=True)

        output_schema = self._get_output_schema()
        # If no entity column, but more than one entity, create an additional column
        if self.entity_augmenter is None and n_entities > 1 and len(self.entities) == 0:
            output_schema = dict(OrderedDict(
                [(_entity_col, VariableType.STR.value)] + list(output_schema.items())))

        for c in result.columns:
            if c != 'entity' and self.fitted_dataset_schema[c] == "int" and c not in self.entities:
                if result[c].dtypes in ["object", "category"]:
                    try:
                        result[c] = to_numeric(result[c].astype(float))
                    except ValueError:
                        result[c] = result[c].astype(str)
                        break
                if result[c].isna().values.any() > 0:
                    output_schema[c] = "float"
                else:
                    result[c] = to_numeric(result[c].astype(int))

        if n_entities is not None and n_entities > 1:
            result = self._anonymize_entities(
                n_entities=n_entities, result=result)

        output_schema, result = self._anonymizer_post_process(
            output_schema, result)

        output_schema, result = self._calculated_features_post_process(
            output_schema, result)

        result_dd = self._pandas_df_to_dask(result)
        dataset = Dataset(result_dd, schema=output_schema)

        if connector:
            try:
                # test if we can connect to the connector
                connector.test()

                self._write_sample_to_connector(dataset, connector, **kwargs)
            except Exception as e:
                print(f'got error while writing to the connector {e}')

        return dataset

    @log_time_factory(logger)
    @synthesizer_sample
    def sample(
            self,
            n_entities: int | None = None,
            smoothing: bool | dict | SmoothingConfig = False,
            fidelity: float | dict | FidelityConfig | None = None,
            sort_result: bool = True,
            condition_on: list[ConditionalFeature] | dict | pdDataFrame | None = None,
            balancing: bool = False,
            random_state: RandomSeed = None,
            connector: BigQueryConnector | ObjectStorageConnector | RDBMSConnector | None = None,
            **kwargs
        ) -> Dataset:
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
        if n_entities is None:
            raise SynthesizerValueError("Please enter a valid integer for the number of synthetic `n_entities` to generate."
                                        "Note: `n_entities` is not the number of rows, as each entity consists of multiple rows. Please provide the number of entities.")

        if n_entities < self.n_entities:
            warn(
                f"The specified value for `n_entities` is less than the number of available entities in the training dataset. "
                f"As a result, the original number of entities ({self.n_entities}) will still be generated.", IgnoredParameter

            )
            n_entities = self.n_entities

        return self._MultiTableSynthesizer__sample(n_entities=n_entities, smoothing=smoothing, fidelity=fidelity,
                             condition_on=condition_on, balancing=balancing, random_state=random_state,
                             connector=connector, **kwargs)

    def _check_sample(self, n_entities: int = 1, smoothing: Union[bool, dict, SmoothingConfig] = False, fidelity: Optional[Union[float, dict, FidelityConfig]] = None, sort_result: bool = True) -> Dataset:
        self._m_get_n_segments_samples = partial(
            self._test_get_n_segments_samples, sample_per_block=n_entities)
        r = self.sample(n_entities, smoothing, fidelity, sort_result)
        self._m_get_n_segments_samples = self._get_n_segments_samples
        return r

    def _anonymize_entities(self, n_entities: int, result: pdDataFrame) -> pdDataFrame:
        """Anonymize newly created entities for TimeSeries.

        This function anonymizes the entity IDs columns. First, we filter the anonymizer configuration
        from the Synthesizer to keep only the entity IDs column. Then, we apply a new anonymizer on these
        columns such that the entities IDs in the synthetic sample follows the proper pattern.

        We cannot re-use the anonymizer from the preprocessing pipeline because by definition, we do not know
        in advance how many entities will be generated via the `sample` method.

        Args:
            n_entities (int): Number of entities in the dataframe `result`.
            result (pdDataFrame): Synthetic sample

        Returns:
            pdDataFrame: Synthetic sample anonymized
        """
        entities_cols = self.entities
        config = {
            k: v for k, v in self.anonymize.get_config().items()
            if any(
                col for col in v.cols
                if col in entities_cols
            )
        }

        builder = AnonymizerConfigurationBuilder(config)

        for v in builder.get_config().values():
            result[v.cols] = v.get_anonymizer()(
                result[v.cols], {e: n_entities for e in v.cols}, **v.params)
        return result
