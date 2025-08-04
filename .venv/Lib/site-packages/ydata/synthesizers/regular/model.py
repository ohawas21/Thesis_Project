"""Synthesizer model."""
from __future__ import annotations

from functools import partial
from os import getenv
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

from pandas import DataFrame as pdDataFrame
from pandas import to_numeric

from ydata.connectors.storages.big_query_connector import BigQueryConnector
from ydata.connectors.storages.object_storage_connector import ObjectStorageConnector
from ydata.connectors.storages.rdbms_connector import RDBMSConnector
from ydata.datascience.common import PrivacyLevel
from ydata.dataset import Dataset
from ydata.dataset.holdout import Holdout
from ydata.metadata.metadata import Metadata
from ydata.preprocessors.identity import Identity as IdentityPreprocessor
from ydata.preprocessors.methods.anonymization import AnonymizerConfigurationBuilder
from ydata.preprocessors.regular.preprocessor import CartHierarchicalPreprocessor, CartHierarchicalSegmentPreprocessor
from ydata.synthesizers.base_model import BaseModel, SegmentByType
from ydata.synthesizers.conditional import ConditionalFeature
from ydata.synthesizers.logger import synthlogger_config
from ydata.synthesizers.prototype import PipelinePrototype
from ydata.synthesizers.utils.models import RegularSynthesizerModel
from ydata.utils.data_types import DataType
from ydata.utils.misc import log_time_factory
from ydata.utils.random import RandomSeed

from ydata._licensing import synthesizer_sample

# Define here the logging status and definition
logger = synthlogger_config(verbose=getenv(
    "VERBOSE", "false").lower() == "true")


class RegularSynthesizer(BaseModel):
    """
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
    """

    __name__ = "RegularSynthesizer"
    _SUPPORTED_DTYPES = [DataType.NUMERICAL, DataType.STR,
                         DataType.CATEGORICAL, DataType.DATE]

    def __init__(
        self,
        *,
        tmppath: Union[str, Path] = None,
        filter_outliers: bool = True,
        strategy: str = "random",
    ):
        super().__init__(tmppath=tmppath)
        self.filter_outliers = filter_outliers
        self._strategy = strategy
        self._holdout = Holdout()

    @property
    def SUPPORTED_DTYPES(self):  # noqa: N802
        return self._SUPPORTED_DTYPES

    @staticmethod
    def _validate_regular_inputs(metadata, holdout, _bypass_min_rows_limit):
        # Holdout validations - only required for the RegularSynthesizer
        if metadata.shape[0] <= BaseModel.MIN_ROWS and not _bypass_min_rows_limit:
            raise NotEnoughRows(
                f"Not enough rows. Training dataset must have at least {BaseModel.MIN_ROWS} rows.")

        cols_with_infinity = [k for k, v in metadata.summary.get(
            'infinity', {}).items() if v > 0]
        if cols_with_infinity:
            msg = "The following columns contain infinite values that are not supported yet: {}".format(
                ','.join(cols_with_infinity))
            raise SynthesizerValueError(msg)

        if 0 > holdout or holdout > 1.:
            raise SynthesizerValueError(
                "Holdout value should be larger or equal to 0 and lower than 1.")

    def _select_pipeline_prototype(
        self, metadata: Metadata, data_types: Optional[dict] = None
    ) -> PipelinePrototype:
        preprocessor = CartHierarchicalSegmentPreprocessor
        if metadata.shape[0] < 5_000 or metadata.shape[1] < 50 or metadata._is_multitable:
            preprocessor = IdentityPreprocessor
        pipeline_def = PipelinePrototype(
            dataset=PipelinePrototype.DatasetScope(
                preprocessor=CartHierarchicalPreprocessor,
                preprocessor_params={"anonymize_config": self.anonymize,
                                     "metadata": metadata},
            ),
            segment=PipelinePrototype.SegmentScope(preprocessor=preprocessor,
                                                   preprocessor_params={
                                                       "metadata": metadata}
                                                   ),
            block=PipelinePrototype.BlockScope(
                synthesizer=RegularSynthesizerModel.CART.value
            ),
        )
        return pipeline_def

    @log_time_factory(logger)
    def fit(
        self,
        X: Dataset,
        metadata: Metadata,
        *,
        condition_on: Union[str, list[str]] | None = None,
        privacy_level: PrivacyLevel | str = PrivacyLevel.HIGH_FIDELITY,
        calculated_features: list[dict[str, str |
                                       Callable | List[str]]] | None = None,
        anonymize: dict | AnonymizerConfigurationBuilder | None = None,
        anonymize_ids: bool = False,
        segment_by: SegmentByType = "auto",
        holdout_size: float = 0.2,
        random_state: RandomSeed = None
    ):
        """
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

        """
        # Transform into list in case user provides a string
        if condition_on is not None and isinstance(condition_on, str):
            condition_on = [condition_on]

        #Validates the provided privacy level
        privacy_level = PrivacyLevel(privacy_level)

        #Validates whether the user inputs are valid
        self._validate_input_values(dataset=X,
                                    metadata=metadata,
                                    condition_on=condition_on, #validates whether the provided condition is valid
                                    segment_by=segment_by, #validates the segmentation decision using the Enum
                                    calculated_features=calculated_features)

        self._validate_regular_inputs(metadata=metadata,
                                      holdout=holdout_size,
                                      _bypass_min_rows_limit=self._bypass_min_rows_limit)

        self.anonymize = anonymize
        logger.debug(
            "[SYNTHESIZER] - Updating the metadata to only keep the columns to be synthesized.")

        self._init_fit(metadata, segment_by, calculated_features,
                       anonymize, condition_on, anonymize_ids=anonymize_ids)

        # Create the train set based on a calculated holdout.
        if (holdout_size > 0.) and len(X) > 10000:
            logger.debug(
                f"[SYNTHESIZER] - Applying the holdout split with a {self._strategy} strategy.")
            train_X, _ = self._holdout.get_split(
                X=X, metadata=metadata, strategy=self._strategy, random_state=random_state
            )
            del X
        else:
            train_X = X

        # Select only the columns to be synthesized
        logger.debug(
            "[SYNTHESIZER] - Selecting only the columns to be synthesized.")
        cols = list(self.data_types.keys())
        train_X = train_X[cols]

        # update metadata
        # if len calculated_features>0
        metadata = metadata[cols]

        # TODO: schema definition inference
        logger.debug("[SYNTHESIZER] - Starting the training procedure.")
        self._fit(train_X, metadata,
                  privacy_level=privacy_level,
                  condition_on=condition_on,
                  holdout_size=holdout_size,
                  random_state=random_state)
        self.is_fitted_ = True
        return self

    def _sample(
        self,
        n_samples: int = 1,
        condition_on: list[ConditionalFeature] | dict | pdDataFrame | None = None,
        balancing: bool = False,
        random_state: RandomSeed = None,
        connector: BigQueryConnector | ObjectStorageConnector | RDBMSConnector | None = None,
        **kwargs
    ):
        """Auxiliar method called by the MultiTableSynth to avoid licensing"""

    def _MultiTableSynthesizer__sample(
        self,
        n_samples: int = 1,
        condition_on: list[ConditionalFeature] | dict | pdDataFrame | None = None,
        balancing: bool = False,
        random_state: RandomSeed = None,
        connector: BigQueryConnector | ObjectStorageConnector | RDBMSConnector | None = None,
        **kwargs
    ):
        """
        Auxiliar method called by the MultiTableSynth to avoid licensing
        """

        result = super()._sample(n_samples, condition_on=condition_on,
                                 balancing=balancing, random_state=random_state)[:n_samples]

        output_schema = self._get_output_schema()

        for c in result.columns:
            if c in output_schema and output_schema[c] == "int":
                if result[c].dtypes in ["object", "category"]:
                    try:
                        result[c] = to_numeric(result[c].astype(float))
                    except ValueError:
                        result[c] = result[c].astype(str)
                        break  # No need to check for empty values
                if result[c].isna().values.any() > 0:
                    output_schema[c] = "float"
                else:
                    result[c] = to_numeric(result[c].astype(int))
            if self.is_multitable and output_schema[c] == "datetime":
                output_schema[c] = "int"

        output_schema, result = self._anonymizer_post_process(
            output_schema, result)

        output_schema, result = self._calculated_features_post_process(
            output_schema, result)

        result_dd = self._pandas_df_to_dask(result)
        dataset = Dataset(result_dd, schema=output_schema)

        logger.info(f'let write into the connector {connector}')
        logger.info(f'using kwargs {kwargs}')
        if connector:
            try:
                # test if we can connect to the connector
                connector.test()

                logger.info('write connector tested successfully')

                self._write_sample_to_connector(dataset, connector, **kwargs)

                logger.info('sample written into connector successfully')
            except Exception as e:
                logger.error(f'got error while writing to the connector {e}')

        return dataset

    @log_time_factory(logger)
    @synthesizer_sample
    def sample(
        self,
        n_samples: int = 1,
        condition_on: list[ConditionalFeature] | dict | pdDataFrame | None = None,
        balancing: bool = False,
        random_state: RandomSeed = None,
        connector: BigQueryConnector | ObjectStorageConnector | RDBMSConnector | None = None,
        **kwargs
    ):
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
        return self._MultiTableSynthesizer__sample(n_samples=n_samples,
                             condition_on=condition_on,
                             balancing=balancing,
                             random_state=random_state,
                             connector=connector,
                             **kwargs)

    @log_time_factory(logger)
    def _check_sample(self,
                      n_samples: int = 1,
                      post_transform: dict[str, Callable] = None,
                      post_filter: dict[str, Callable[[Any], bool]] = None) -> Dataset:
        self._m_get_n_segments_samples = partial(
            self._test_get_n_segments_samples, sample_per_block=n_samples)
        r = self.sample(n_samples, post_transform, post_filter)
        self._m_get_n_segments_samples = self._get_n_segments_samples
        return r

