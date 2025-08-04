"""Base model gfor ydata synthesizers."""
import os

import uuid
from collections import OrderedDict
from copy import deepcopy
from math import ceil
from os import getenv
from pathlib import Path, PurePath
from shutil import copy as shcopy
from typing import Callable, Literal
from warnings import warn

import numpy as np
import pandas as pd
from dask.dataframe import concat as ddConcat
from dask.dataframe.core import DataFrame as ddDataFrame
from dask.dataframe.io import from_pandas as dd_from_pandas

from dill import dump, load
from numpy import arange as nparange
from numpy import isclose as npisclose
from pandas import NA
from pandas import DataFrame as pdDataFrame
from pandas import Series as pdSeries
from pandas import concat, unique

from ydata.characteristics import ColumnCharacteristic
from ydata.characteristics.characteristics import deduce_anonymizer_config_for_PII, deduce_anonymizer_config_for_STR
from ydata.connectors.storages.big_query_connector import BigQueryConnector
from ydata.connectors.storages.databricks_lakehouse import DatabricksLakehouse
from ydata.connectors.storages.object_storage_connector import ObjectStorageConnector
from ydata.connectors.storages.rdbms_connector import RDBMSConnector
from ydata.datascience.common import PrivacyLevel
from ydata.dataset import Dataset
from ydata.dataset.dataset_type import DatasetType
from ydata.dataset.settings import RUNNING_ENV
from ydata.dataset.utils import humanize_dtypes
from ydata.metadata import Metadata
from ydata.metadata.column import Column
from ydata.preprocessors.base import Preprocessor
from ydata.preprocessors.methods.anonymization import AnonymizerConfigurationBuilder, AnonymizerType
from ydata.preprocessors.preprocess_methods import AnonymizerEngine, Gaussianization
from ydata.synthesizers.calculated_features import CalculatedFeature, _validate_calculated_features
from ydata.synthesizers.conditional import ConditionalFeature, ConditionalUtils
from ydata.synthesizers.entity_augmenter import EntityAugmenter, FidelityConfig, SmoothingConfig
from ydata.synthesizers.exceptions import (SynthesizerValueError, MissingDeviceException, NotFittedException, SegmentationStrategyException,
                                           TemporaryPathException, SynthesizerAssertionError)
from ydata.synthesizers.logger import synthlogger_config
from ydata.synthesizers.privacy import DifferentialPrivacyLayer
from ydata.synthesizers.prototype import DataBlock, DataSegment, PipelinePrototype
from ydata.synthesizers.utils.segmentation import (EntitySegmentation, NNSegmentation, NoSegmentation,
                                                   SegmentationStrategy, SegmentInfo, SimpleSegmentation)
from ydata.synthesizers.utils.segmentation import Strategy as SegStrategy
from ydata.synthesizers.utils.slicing import NoSlicing, SimpleSlicing, SliceInfo, SlicingStrategy
from ydata.utils.data_types import DataType, VariableType
from ydata.utils.exceptions import LessRowsThanColumns, NotEnoughRows, SmallTrainingDataset
from ydata.utils.misc import log_time_factory
from ydata.utils.random import RandomSeed
from ydata.utils.dask import DaskCluster
from ydata._licensing import licensed

from ydata.utils.logger import DATATYPE_MAPPING, SDKLogger

logger = synthlogger_config(verbose=getenv(
    "VERBOSE", "false").lower() == "true")

metrics_logger = SDKLogger(name="Metrics logger")


SegmentByType = bool | str | list[str] | Literal["auto"] | SegStrategy.to_literal(
) | SegStrategy | SegmentationStrategy | None


def get_default_tmppath(folder: Path = None) -> Path:
    if folder:
        return folder
    elif os.name == "posix" and RUNNING_ENV == "LOCAL":
        return Path("/tmp")
    else:
        return Path("./")

class BaseModel:
    """Base class for the synthesis models."""

    is_fitted_ = False

    DEFAULT_LAB_TMP = ".local_models"
    MIN_ROWS = 50
    LOW_ROWS = 100

    @licensed
    def __init__(self, tmppath: str | Path = None):
        self.calculated_features = []
        self.pipelines = {}
        self._pipeline_def: PipelinePrototype | None = None
        self.segment_by: SegmentByType = "auto"
        self.dataset_preprocessor: Preprocessor | None = None
        self.features_order = None
        self.data_types = {}
        self.fitted_dataset_schema = {}
        self.segmenter: SegmentationStrategy | None = None
        self.slicer: SlicingStrategy | None = None
        self.segmentation_strategy = None
        self.slicing_strategy = None
        self.dataset_type = None
        self.uuid = uuid.uuid4()
        self.tmppath: Path = (
            Path(tmppath)
            if tmppath is not None
            else get_default_tmppath(Path(BaseModel.DEFAULT_LAB_TMP) / str(self.uuid))
        )
        self.anonymize: dict | None = None
        self._constant_features = {}
        self._unique_features = {}
        self.entity_augmenter = None
        self.pivot_columns = None
        self.entities_type = {}
        self._m_get_n_segments_samples = self._get_n_segments_samples
        self._privacy_level = None
        self._condition_on = None
        self.time_series = False
        self.metadata_summary = None
        self.categorical_vars = None
        self._bypass_min_rows_limit: bool = False
        self.random_state = None

        try:
            if not self.tmppath.exists():
                self.tmppath.mkdir(parents=True)
        except Exception as e:
            msg = f'Could not create temporary path "{self.tmppath}"!'
            logger.error(msg)
            logger.error(e)
            raise TemporaryPathException(msg)

    @property
    def privacy_level(self):
        return self._privacy_level

    @property
    def anonymized_columns(self) -> list[str]:
        if self.anonymize is None:
            return []
        return AnonymizerEngine.get_anonymized_columns(self.anonymize)

    def _select_pipeline_prototype(
        self, metadata: Metadata, data_types: dict | None = None
    ) -> PipelinePrototype:
        """Select or construct the pipeline prototype to be used to instantiate
        the synthesizer flow.

        This method should receive all the necessary contextual information to determine the best
        preprocessors and the best model.

        Args:
            metadata (Metadata): Metadata of the dataset to synthesize
            data_types (int | str): Data types which are to be synthesized

        Returns:
            PipelinePrototype: Pipeline prototype to use to build the synthesizer
        """
        raise NotImplementedError

    @log_time_factory(logger)
    def fit(
        self,
        X,
        y=None,
        segment_by: SegmentByType = "auto",
        anonymize: dict | None = None,
        random_state: RandomSeed = None
    ):
        raise NotImplementedError

    @log_time_factory(logger)
    def sample(self, n_samples: int | None = None,
                     n_entities: int | None = None) -> Dataset:
        if not self.is_fitted_ :
            raise Exception("Synthesizer model is untrained. Please call the 'fit' method with your data before attempting to sample.")

    def _get_entity_batch_numbers(self, n_samples: int | None) -> int:
        """Determine the batch number based on the number of entities.

        The batch number is always 1 except for multi-entities dataset
        where it can be larger.
        """
        from ydata.synthesizers import TimeSeriesSynthesizer
        entity_batch_numbers = 1
        if (self.dataset_type == DatasetType.TIMESERIES or isinstance(self, TimeSeriesSynthesizer)) and len(self.entities) > 0 and self.entity_augmenter and n_samples > self.entity_augmenter.n_entities:
            entity_batch_numbers = int(
                n_samples // self.entity_augmenter.n_entities)
            if n_samples % self.entity_augmenter.n_entities != 0:
                entity_batch_numbers += 1
        return entity_batch_numbers

    def _test_get_n_segments_samples(self, segment: SegmentInfo, n_samples: int | None = None, sample_per_block: int = 1) -> int:
        """Used by integration to check if a synthesizer can sample by
        requesting some records per block."""
        return sample_per_block

    def _get_n_segments_samples(self, segment: SegmentInfo, n_samples: int | None = None) -> int:
        n_samples = n_samples if n_samples is not None else self.nrows_training
        n_segment_samples = int(ceil(n_samples * segment.info.ratio))
        if self.dataset_type == DatasetType.TIMESERIES:
            if len(self.entities) > 0:
                n_segment_samples = segment.info.size
            else:
                n_segment_samples = int(
                    min(segment.info.size, n_segment_samples))
        return n_segment_samples

    @log_time_factory(logger)
    def _apply_constant_columns(self, X: ddDataFrame) -> ddDataFrame:
        """
        Method that creates and concat a DASK dataframe with the constant columns
        Args:
            X: The generated synthetic dataset through the generative models

        Returns: A DASK dataframe
        """
        if bool(self._constant_features):
            df = pd.DataFrame([self._constant_features] * len(X.index))
            df_dd = dd_from_pandas(df, npartitions=X.npartitions)
            X = X.merge(df_dd, left_index=True, right_index=True, how='left')
        return X

    @log_time_factory(logger)
    def _generate_entities_ids(self, entity_batch_numbers: int, batch_id: int, batch_entities: list, n_samples: int, X: ddDataFrame, bypass_entities_anonymization: list | None = None) -> ddDataFrame:
        if len(self.entities) > 0 and self.entity_augmenter is not None and batch_entities is not None:
            bypass_entities_anonymization = bypass_entities_anonymization if bypass_entities_anonymization is not None else []
            # Generate entities IDs
            for c in self.entity_augmenter.entities:
                if c not in bypass_entities_anonymization:
                    if self.entities_type[c] == VariableType.STR:
                        batch_entity_ids = [
                            f'{c}_{batch_id * len(batch_entities) + i}' for i in range(len(batch_entities))]
                    else:
                        batch_entity_ids = [
                            batch_id * len(batch_entities) + i for i in range(len(batch_entities))]
                    if batch_id == entity_batch_numbers - 1 and n_samples % self.entity_augmenter.n_entities != 0:
                        batch_entity_ids = batch_entity_ids[:(
                            n_samples % self.entity_augmenter.n_entities)]
                    X[c] = X[c].map(
                        dict(zip(batch_entities, batch_entity_ids)))

                    # Remove the extra-entities
                    if batch_id == entity_batch_numbers - 1 and n_samples % self.entity_augmenter.n_entities != 0:
                        X = X[X[c].isin(batch_entity_ids)]
                else:
                    # Remove the extra-entities
                    if batch_id == entity_batch_numbers - 1 and n_samples % self.entity_augmenter.n_entities != 0:
                        X = X[X[c].isin(batch_entities[:n_samples %
                                        self.entity_augmenter.n_entities])]
        return X

    @log_time_factory(logger)
    def _map_close_to_zero(self, X: ddDataFrame) -> ddDataFrame:
        output_dtypes = self.dataset_preprocessor.output_dtypes(-1)
        for k, v in output_dtypes.items():
            if v.datatype == DataType.NUMERICAL and v.vartype == VariableType.FLOAT and X[k].dtype != "O":
                X[k] = X[k].apply(
                    lambda x: x if not npisclose(x, 0.0) else 0.0
                )
        return X

    @log_time_factory(logger)
    def _get_block_sample(self,
                          block_id: int,
                          block: DataBlock,
                          n_segment_samples: int,
                          smoothing: bool | dict | SmoothingConfig = False,
                          fidelity: float | dict | FidelityConfig | None = None,
                          bootstrapping_df: pdDataFrame | None = None,
                          random_state: RandomSeed = None) -> pdDataFrame:
        if len(self.entities) > 0 and self.entity_augmenter is not None:
            samples = self.entity_augmenter._sample_new_entities(
                str(block_id),
                block.synthesizer,
                block.preprocessor,
                smoothing=smoothing,
                fidelity=fidelity,
                bootstrapping=bootstrapping_df, random_state=random_state)
        else:
            samples = block.synthesizer.sample(n_samples=int(n_segment_samples),
                                               bootstrapping=bootstrapping_df, random_state=random_state)
        return samples

    @log_time_factory(logger)
    def _batch_postprocess(self, entity_batch_numbers: int, batch_id: int, batch_entities: list | None, n_samples: int, X: ddDataFrame, bypass_entities_anonymization: list[str] | None = None) -> ddDataFrame:
        inv_X = self.dataset_preprocessor.inverse_transform(
            X.copy()) if not self.is_multitable else X
        if len(self.entities) > 0 and self.entity_augmenter is not None:
            ent = [c for c in self.entities if c not in bypass_entities_anonymization]
            inverse_cols = [c for c in X.columns if c not in ent]
            X[inverse_cols] = inv_X[inverse_cols]
        else:
            X = inv_X
        X = self._apply_constant_columns(X)
        if batch_entities is not None:
            X = self._generate_entities_ids(
                entity_batch_numbers, batch_id, batch_entities, n_samples, X, bypass_entities_anonymization=bypass_entities_anonymization)
        return X

    @log_time_factory(logger)
    def _batch_alignment(self, batch_result: list[pdDataFrame]) -> list[pdDataFrame]:
        """Realign the segment columns.

        In case of multi-table synthetiszer, it is possible that
        segments will not have the same number of columns because a
        segment might have NaN values and another not. The first segment
        would have an additional column to flag the NaN values. Because
        of this, we need to realign the dimensions with the assumption
        that only additional flag columns are added which is why the
        extra columns are added with a flag to 0.
        """
        all_columns = []
        for i, batch in enumerate(batch_result):
            all_columns.extend(list(batch.columns))
        all_columns = list(set(all_columns))
        for i, batch in enumerate(batch_result):
            for c in all_columns:
                if c not in batch.columns:
                    batch_result[i][c] = 0
            batch_result[i] = batch_result[i].loc[:, ~
                                                  batch_result[i].columns.duplicated()].copy()
            batch_result[i] = batch_result[i][sorted(all_columns)]
        return batch_result

    @log_time_factory(logger)
    def _load_fitted_block(self, block_file):
        with open(block_file, 'rb') as dump_file:
            block = load(dump_file)
        return block

    @log_time_factory(logger)
    def _sample(self, n_samples: int | None = None,
                smoothing: bool | dict | SmoothingConfig = False,
                fidelity: float | dict | FidelityConfig | None = None,
                condition_on: list[ConditionalFeature] | pdDataFrame | dict | None = None,
                balancing: bool = False,
                random_state: RandomSeed = None,
                bypass_entities_anonymization: list[str] | None = None) -> pdDataFrame:
        # Allow reset the seed using None
        self._check_is_fitted()
        bypass_entities_anonymization = bypass_entities_anonymization if bypass_entities_anonymization is not None else []
        logger.info('[SYNTHESIZER] - Start generating model samples.')

        logger.info('[SYNTHESIZER] - Init Dask cluster for sampling.')
        cluster = DaskCluster()

        if n_samples is None:
            n_samples = 1
            if len(self.entities) > 0:
                n_samples = self.entity_augmenter.n_entities


        if balancing and self._condition_on:
            categorical_cond_vars = [
                c for c in self._condition_on if c in self.categorical_vars]
            if categorical_cond_vars:
                condition_on = condition_on if condition_on is not None else {}
                if isinstance(condition_on, dict):
                    for cat_var in categorical_cond_vars:
                        condition_on[cat_var] = {
                            "balancing": True
                        }

        if condition_on is not None:
            if len(condition_on) > 0:
                condition_on = ConditionalUtils.prepare_conditional_sample(
                    condition_on=condition_on,
                    conditional_features=self._condition_on,
                    data_types=self.data_types,
                    n_samples=max(
                        self.nrows_training if self.time_series else n_samples, len(self.pipelines)),
                    preprocessor=self.dataset_preprocessor if not self.is_multitable else None,
                    metadata_summary=self.metadata_summary,
                    random_state=random_state
                )
            else:
                condition_on = None

        # The entity augmentation is based on the existing entities which means that
        # a generated sample can only have at most a number of entities equals to the number
        # of entities in the training set.
        # It is done this way because it is the only way to generate new entities that
        # can keep the same interactions and dependences with **all** the other entities.
        #
        # Therefore, we introduce a `entity_batch` where each batch contains exactly as many
        # entities as the training dataset.
        entity_batch_numbers = self._get_entity_batch_numbers(n_samples)
        result = []
        result_nrows = 0
        for batch_id in range(entity_batch_numbers):
            if entity_batch_numbers > 1:
                logger.info(f'[SYNTHESIZER] - Sample batch number {batch_id}')
            block_id = 0
            batch_result = []
            sum_n_segment_samples = 0
            for segment_name, segment in self.pipelines.items():
                if len(self.pipelines) > 1:
                    logger.info(
                        f'[SYNTHESIZER] - Sample segment {segment_name}')
                blocks_samples = []
                n_segment_samples = self._m_get_n_segments_samples(
                    segment, n_samples if self.dataset_type != DatasetType.TIMESERIES else None)

                bootstrapping_df = None
                if condition_on is not None:
                    if self.entity_augmenter is not None:
                        n_segment_samples = condition_on.shape[0]
                    # Because of the rounding, sometimes the last segment goes over the limit.
                    if sum_n_segment_samples + n_segment_samples >= condition_on.shape[0]:
                        n_segment_samples = condition_on.shape[0] - \
                            sum_n_segment_samples

                    bootstrapping_df = condition_on. \
                        iloc[sum_n_segment_samples:sum_n_segment_samples+n_segment_samples]. \
                        reset_index(drop=True)

                if n_segment_samples == 0:
                    continue

                for block_name, block_file in segment.blocks.items():
                    if len(segment.blocks) > 1:
                        logger.info(
                            f'[SYNTHESIZER] - Sample block {block_name}')
                    block = self._load_fitted_block(block_file)

                    bootstrapping_block = None
                    if bootstrapping_df is not None:
                        bootstrapping_cols = [
                            c for c in block.info.mask if c in bootstrapping_df.columns]
                        if bootstrapping_cols:
                            bootstrapping_block = bootstrapping_df[bootstrapping_cols]
                            bootstrapping_block = block.preprocessor.transform(
                                Dataset(bootstrapping_block)).to_pandas()

                    block_samples = self._get_block_sample(
                        block_id, block, n_segment_samples,
                        smoothing, fidelity, bootstrapping_block, random_state=random_state)
                    blocks_samples.append(block_samples)
                    block_id += 1

                # Segment postprocess
                sum_n_segment_samples += n_segment_samples
                segment_samples = concat(blocks_samples, axis=1)
                segment.preprocessor.inverse_transform(segment_samples)

                batch_result.append(segment_samples)

                # Early stopping for the non-time series synthesizer
                if self.dataset_type != DatasetType.TIMESERIES and sum_n_segment_samples > n_samples:
                    break

            # Batch postprocess
            batch_result = self._batch_alignment(batch_result)
            batch_result = concat(batch_result, ignore_index=True)
            batch_result.reset_index(drop=True, inplace=True)

            # Column duplicates might appear with Gaussianization and Time Series as the sort-by-key columns
            # must be copied to each slice.
            batch_result = batch_result.loc[:,
                                            ~batch_result.columns.duplicated()]

            result_nrows += batch_result.shape[0]
            batch_entities = []
            if len(self.entities) > 0 and self.entity_augmenter is not None:
                for c in self.entity_augmenter.entities:
                    if c in bypass_entities_anonymization:
                        batch_entities += self.metadata_summary['value_counts'][c].index.to_list(
                        )
                    else:
                        batch_entities += unique(
                            batch_result[c].values.ravel('K')).tolist()
            if len(batch_entities) == 0:
                batch_entities = None

            batch_result=batch_result.infer_objects()
            batch_result = self._pandas_df_to_dask(batch_result)

            logger.info('[SYNTHESIZER] - Postprocessing.')
            batch_result = self._batch_postprocess(
                entity_batch_numbers, batch_id, batch_entities, n_samples, batch_result, bypass_entities_anonymization)
            result.append(batch_result)

        # Final sample postprocess
        result = ddConcat(result, axis=0)
        result = result.persist(optimize_graph=True)

        #This is the most costly function of the process.
        """
        logger.info('[SYNTHESIZER] - Map close to zero.')
        #result = self._map_close_to_zero(result)
        logger.info('[SYNTHESIZER] - Map close to zero - finished')
        """

        cf_features = list(
            set([f for cf in self.calculated_features for f in cf.features]))
        unique_assign_kwargs = {}

        if self._anonymize_ids:
            for col_name, col in self._unique_features.items():
                if col_name not in cf_features:
                    col_id_vals = nparange(result_nrows, dtype=int)
                    if col.vartype == VariableType.STR:
                        col_id_vals = [f'{col_name}_{x}' for x in col_id_vals]
                    unique_assign_kwargs[col_name] = pdSeries(col_id_vals)
            if unique_assign_kwargs:
                result = result.assign(**unique_assign_kwargs)

        # Computation to Pandas is performed here!
        return result.compute().reset_index(drop=True)

    @log_time_factory(logger)
    def _pandas_df_to_dask(self, df: pdDataFrame) -> ddDataFrame:
        segment_size: int = int(list(self.pipelines.values())[0].info.size)
        return dd_from_pandas(df, chunksize=segment_size)

    @log_time_factory(logger)
    def _init_fit(
        self,
        metadata: Metadata,
        segment_by: SegmentByType = "auto",
        calculated_features: list[dict[str, str |
                                       Callable | list[str]]] | None = None,
        anonymize: dict | AnonymizerConfigurationBuilder | None = None,
        condition_on: list[str] | None = None,
        anonymize_ids: bool = False
    ) -> Metadata:
        """Initialize the fit method.

        Returns:
            Metadata: Modified metadata with only the columns to be synthesized
        """
        anonymize = anonymize if anonymize is not None else {}
        if isinstance(anonymize, dict):
            builder = AnonymizerConfigurationBuilder(anonymize)
        else:
            builder = anonymize

        self._anonymize_ids = anonymize_ids

        if metadata.dataset_attrs:
            self._unique_features = {
                w.column: metadata.columns[w.column]
                for w in metadata.warnings["unique"] if w.column not in metadata.dataset_attrs.entities
                and ColumnCharacteristic.ID in metadata.columns[w.column].characteristics
            }
        else:
            self._unique_features = {
                w.column: metadata.columns[w.column]
                for w in metadata.warnings["unique"] if ColumnCharacteristic.ID in metadata.columns[w.column].characteristics
            }

        str_anonymizer: dict = deduce_anonymizer_config_for_STR(metadata)
        if str_anonymizer:
            builder.add_config(str_anonymizer)

        if anonymize_ids:
            # Automatically anonymize ID if not unique and no other way to anonymize already specified
            ids_anonymizer = {k: AnonymizerType.INT for k, col in metadata.columns.items()
                              if ColumnCharacteristic.ID in col.characteristics
                              and k not in self._unique_features
                              and k not in anonymize
                              }
            if ids_anonymizer:
                logger.warning("[SYNTHESIZER] - The following ID columns will be automatically anonymized: {}".format(
                    ', '.join(ids_anonymizer.keys())))
                builder.add_config(ids_anonymizer)

        pii_anonymizer = deduce_anonymizer_config_for_PII(metadata)
        pii_anonymizer = {k: v for k, v in pii_anonymizer.items()
                          if k not in str_anonymizer}
        if pii_anonymizer:
            builder.add_config(pii_anonymizer)

        self.anonymize = builder

        self.segment_by = segment_by
        self.dataset_type = metadata.dataset_type
        self.nrows_training = metadata.shape[0]
        self.entities = metadata.dataset_attrs.entities if metadata.dataset_attrs is not None else []
        self.metadata_summary = metadata.summary
        self.categorical_vars = metadata.categorical_vars
        self.is_multitable = metadata._is_multitable

        if condition_on is not None:

            ConditionalUtils.validate_conditional_features(
                condition_on=condition_on,
                dataset_columns=list(metadata.columns.keys()),
                anonymize_columns=self.anonymized_columns if not self.is_multitable else None,
                dataset_attrs=metadata.dataset_attrs
            )

        if calculated_features is not None:
            self.calculated_features = [
                CalculatedFeature.from_dict(cf) for cf in calculated_features
            ]
        else:
            self.calculated_features = []

        self.features_order = OrderedDict(
            {
                col: v
                for col, v in metadata.columns.items()
                if v.datatype in self.SUPPORTED_DTYPES
            }
        )

        self.data_types = self._get_filtered_columns_dtypes(
            self.features_order)
        self.fitted_dataset_schema = self._get_fitted_dataset_schema(metadata)
        logger.info(
            f"[SYNTHESIZER] - Number columns considered for synth: {len(self.data_types.items())}"
        )

        if self._pipeline_def is None:
            self._pipeline_def = self._select_pipeline_prototype(
                metadata=metadata)

    @log_time_factory(logger)
    def __get_block_path(self, segment_name, slice_name):
        return Path(self.tmppath) / f"{self.uuid}_{segment_name}_{slice_name}.pkl"

    @log_time_factory(logger)
    def _get_output_schema(self):
        output_schema = {
            k: v.vartype.value for k, v in self.features_order.items()
        }
        if self.entity_augmenter is not None:
            for c in self.entity_augmenter.entities:
                if c in output_schema.keys():
                    output_schema[c] = self.entities_type[c].value
        return output_schema

    @log_time_factory(logger)
    def _save_fitted_block(self, block, block_synth_file):
        with open(block_synth_file, 'wb') as dump_file:
            dump(block, dump_file)

    def _is_feature_valid(self, feature: str):
        if feature in self._constant_features:
            return False
        if self._anonymize_ids and feature in self._unique_features:
            return False
        return True

    @log_time_factory(logger)
    def _fit(self, X: Dataset, metadata: Metadata, n_entities: int = None,
             privacy_level: PrivacyLevel | None = None,
             condition_on: list[str] | None = None,
             fit_params: dict | None = None,
             holdout_size: float = 0.2,
             random_state: RandomSeed = None):
        """Generic fit method to execute a Pipeline Prototype according to a
        Block strategy.

        Args:
            X (Dataset): Training data
            metadata (Metadata): Metadata corresponding to the training data
            privacy_level (PrivacyLevel): Privacy level
            condition_on (list[str]): List of conditional features
            fit_params: Additional parameters to pass to the fit method of the block synthesizer instance
            holdout_size: % of the data that is used as holdout. Default to 0. as it is not used by all synthesizers.
            random_state: random generator or seed for the synthesizer fit
        """

        metrics_logger.info(dataset=X,
                            datatype=DATATYPE_MAPPING[self.__class__.__name__],
                            method='synthesizer')

        if metadata.shape[0] < BaseModel.LOW_ROWS:
            warn(
                f"Small training dataset detected. For optimal results, training data should have at least {BaseModel.LOW_ROWS} rows.", SmallTrainingDataset)

        if metadata.shape[0] < metadata.shape[1]:
            warn("Training data has less rows than columns. This might lead to overfitting or degraded results.", LessRowsThanColumns)

        if fit_params is None:
            fit_params = {}

        self._condition_on = condition_on

        constant_cols = metadata.isconstant

        entities = [] if metadata.dataset_attrs is None else metadata.dataset_attrs.entities

        for col in constant_cols:
            if col not in entities:
                if metadata.columns[col].vartype in [VariableType.FLOAT, VariableType.DATE, VariableType.DATETIME]:
                    val = metadata.summary['domains'][col]['min']
                elif metadata.summary["value_counts"][col].empty:
                    if col in metadata.numerical_vars:
                        val = np.nan
                    elif col in metadata.date_vars:
                        val = np.datetime64('NaT')
                    else:
                        val = NA
                else:
                    val = metadata.summary["value_counts"][col].index.values[0]
                self._constant_features[col] = val

        input_dtypes = deepcopy(metadata.columns)
        input_dtypes = {
            k: v
            for k, v in input_dtypes.items()
            if self._is_feature_valid(k)
        }

        logger.debug("[SYNTHESIZER] - Dropping the constant features.")
        X.drop_columns(columns=list(
            self._constant_features.keys()), inplace=True)

        params = self._pipeline_def.dataset.preprocessor_params
        self.dataset_preprocessor = self._pipeline_def.dataset.preprocessor(
            **params)

        logger.debug("[SYNTHESIZER] - Executing the preprocessing steps.")
        X_processed = self.dataset_preprocessor.fit_transform(
            X=X,
            input_dtypes=input_dtypes,
            anonymization__metadata=metadata,
            anonymization__config=self.anonymize,
        )
        X_processed = X_processed.to_pandas()

        self.time_series = n_entities is not None

        if self.time_series:
            X_processed = X_processed.reset_index(
                drop=True).loc[X.sorted_index(by=self.sortbykey)]

        X_processed.reset_index(drop=True, inplace=True)
        dataset_input_dtypes = self.dataset_preprocessor.output_dtypes(-1)

        # Applying the privacy layer to the processed dataset.
        self._privacy_level = privacy_level \
            if privacy_level is not None \
            else PrivacyLevel.HIGH_FIDELITY
        privacy_layer = DifferentialPrivacyLayer(
            time_series=self.time_series, random_state=self.random_state)
        X_processed = privacy_layer.apply(
            X=X_processed[dataset_input_dtypes.keys()],
            privacy_level=self._privacy_level,
            input_dtypes=dataset_input_dtypes)

        metadata.summary["nrows"] = len(X_processed)
        if n_entities is not None:
            n_entities = n_entities if isinstance(n_entities, (int, np.integer)) else int(n_entities.compute())

        self.segmenter, self.slicer = self._select_block_strategy(
            X_processed, metadata, dataset_input_dtypes
        )
        logger.info(
            f"[SYNTHESIZER] - Starting the synthetic data modeling process over "
            f"{self.segmenter.n_segments}x{self.slicer.n_slices} blocks."
        )

        if self._condition_on is not None:
            self.pivot_columns = self._condition_on.copy()
        else:
            self.pivot_columns = []
        # Pivot columns are the columns that will be used to boostrap the augmentation.
        # The columns which are not pivot will be generated using the synthesizer model learnt over the training set.
        if len(self.entities) > 0 and n_entities > 1:
            self.pivot_columns += [c for c, k in input_dtypes.items(
            ) if k.datatype == DataType.NUMERICAL and k.vartype == VariableType.FLOAT and c not in self.sortbykey and c not in self.pivot_columns]
            self.entity_augmenter = EntityAugmenter(
                X_processed, metadata, n_entities, self.pivot_columns)

        block_id = 0
        for segment in self.segmenter:
            if self.segmenter.n_segments > 1:
                logger.info(
                    f"[SYNTHESIZER] - Generating pipeline for segment {segment.name}"
                )
            X_segment, segment_info = self._preprocess_segment(
                segment_info=segment,
                X=X_processed[segment.mask][dataset_input_dtypes.keys()],
                metadata=metadata,
                dtypes=dataset_input_dtypes,
            )
            self.pipelines[segment.name] = segment_info
            segment_input_dtypes = segment_info.preprocessor.output_dtypes(-1)
            self.slicer = self._select_slicing_strategy(
                X_segment, metadata, segment_input_dtypes
            )
            for slice_ in self.slicer:
                if self.slicer.n_slices > 1:
                    logger.info(
                        f"[SYNTHESIZER] - Generating pipeline for block {slice_.name}"
                    )
                block = self._fit_block(
                    block_id=str(block_id),
                    block_info=slice_,
                    X=X_segment[slice_.mask],
                    metadata=metadata[slice_.mask],
                    dtypes={
                        k: v
                        for k, v in segment_input_dtypes.items()
                        if k in slice_.mask
                    },
                    fit_params=fit_params,
                    random_state=self.random_state
                )
                block_synth_file = self.__get_block_path(
                    segment.name, slice_.name)

                self._save_fitted_block(block, block_synth_file)
                self.pipelines[segment.name].blocks[slice_.name] = block_synth_file
                block_id += 1

    @log_time_factory(logger)
    def _preprocess_segment(
        self,
        segment_info: SegmentInfo,
        X: pdDataFrame,
        metadata: Metadata,
        dtypes: dict[str, Column],
    ) -> tuple[pdDataFrame, DataSegment]:
        """Generic fit method to execute a Pipeline Prototype according to a
        Block strategy.

        Args:
            segment_info (SegmentInfo): Segment description
            X (pdDataFrame): Segment data
            metadata (Metadata): Metadata corresponding to the segment data

        Returns:
            pdDataframe: Preprocessed segment data
            DataSegment: Concrete data segment information
        """
        params = self._pipeline_def.segment.preprocessor_params
        preprocessor = self._pipeline_def.segment.preprocessor(**params)
        logger.info("[SYNTHESIZER] - Preprocess segment")
        X_ = preprocessor.fit_transform(X, input_dtypes=dtypes)
        return X_, DataSegment(preprocessor=preprocessor, info=segment_info)

    @log_time_factory(logger)
    def _fit_block(self,
                   block_id: str,
                   block_info: SliceInfo,
                   X: pdDataFrame,
                   metadata: Metadata,
                   dtypes: dict[str, Column],
                   fit_params: dict | None,
                   random_state: RandomSeed = None) -> DataBlock:
        """Generic fit method to execute a Pipeline Prototype according to a
        Block strategy.

        Args:
            block_info (SliceInfo): Block description
            X (pdDataFrame): Block data
            metadata (Metadata): Metadata corresponding to the block data

        Returns:
            DataBlock: Concrete data block information
        """
        preprocessor = self._pipeline_def.block.preprocessor()
        params = self._pipeline_def.block.synthesizer_params
        synthesizer = self._pipeline_def.block.synthesizer(
            random_state=random_state, *params)
        logger.info("[SYNTHESIZER] - Synthesizer init.")
        logger.info(
            "[SYNTHESIZER] - Processing the data prior fitting the synthesizer."
        )
        X_ = preprocessor.fit_transform(X, input_dtypes=dtypes)
        input_dtypes = preprocessor.output_dtypes(-1)
        synthesizer.fit(X=X_, metadata=metadata, dtypes=input_dtypes,
                        bootstrapping_cols=self.pivot_columns, **fit_params)
        if self.entity_augmenter is not None:
            self.entity_augmenter.fit_block_bootstraper(block_id, X_)

        return DataBlock(synthesizer=synthesizer, preprocessor=preprocessor, info=block_info)

    @property
    def SUPPORTED_DTYPES(self):  # noqa: N802
        raise NotImplementedError

    @property
    def _device(self):
        if self._pipeline_def is None:
            raise MissingDeviceException(
                "Synthesizer was not yet picked. No recommended acceleration device was set."
            )
        return self._pipeline_def.device

    def _init_metadata(self, dataset: Dataset, metadata: Metadata):
        if metadata is None:
            raise NotImplementedError

        assert isinstance(
            metadata, Metadata
        ), "Please provide a valid Metadata instance."
        return metadata

    def _get_filtered_columns_dtypes(self, columns: dict) -> dict:
        _calculated_features = []
        for c in self.calculated_features:
            _calculated_features.extend(c.features)

        features = {
            k: v.datatype
            for k, v in columns.items()
            if v.datatype
            if k not in _calculated_features
            and self._is_feature_valid(k)
        }

        return features

    def _get_fitted_dataset_schema(self, metadata: Metadata) -> dict:
        features = {k: v.vartype.value for k, v in metadata.columns.items(
        ) if v.datatype in self.SUPPORTED_DTYPES and k not in self._constant_features}
        return features

    @log_time_factory(logger)
    def _apply_calculated_features(self, dataframe: pdDataFrame) -> pdDataFrame:
        for calculated_feature in self.calculated_features:
            if len(calculated_feature.features) > 1:
                features = calculated_feature.features
            else:
                features = calculated_feature.features[0]
            dataframe[features] = calculated_feature.apply_to(
                dataframe
            )
        return dataframe

    @log_time_factory(logger)
    def _select_block_strategy(
        self, X: Dataset, metadata: Metadata, dtypes: dict[str, Column]
    ) -> tuple[SegmentationStrategy, SlicingStrategy]:
        """Generic method to select a Block strategy.

        Args:
            X (Dataset): dataset to be broken down into blocks
            metadata (Metadata): Metadata corresponding to the dataset
            dtypes: (dict[str, Column]) Dataset schema

        Returns:
            Tupltuplee[SegmentationStrategy, SlicingStrategy]: Block strategy
        """
        segmenter = self._select_segmentation_strategy(
            X, metadata, dtypes, self.segment_by
        )
        slicer = self._select_slicing_strategy(X, metadata, dtypes)
        return segmenter, slicer

    @staticmethod
    @log_time_factory(logger)
    def _select_segmentation_strategy(
        X: Dataset,
        metadata: Metadata,
        dtypes: dict[str, Column],
        strategy: SegmentByType = "auto",
    ) -> SegmentationStrategy:
        """Select the segmentation strategy to use.

        If a strategy instance or type of strategy is specified, this will bypass the generic flow.

        Args:
            X (Dataset): dataset to be segmented
            metadata (Metadata): Metadata corresponding to the dataset
            dtypes: (dict[str, Column]) Dataset schema
            strategy: Hint to determine the segmenter to be used

        Returns:
            SegmentationStrategy: Selected segmentation strategy
        """

        def auto_strategy(
            X: Dataset, metadata: Metadata, dtypes: dict[str, Column]
        ) -> SegmentationStrategy:
            """Define the segmentation strategy for the 'auto' value."""
            segmenter = None
            if X.shape[0] < 30_000:
                segmenter = NoSegmentation(X, metadata, dtypes)
            else:
                segmenter = SimpleSegmentation(X, metadata, dtypes)
            return segmenter

        def strategy_name_to_instance(
            name, X: Dataset, metadata: Metadata, dtypes: dict[str, Column]
        ) -> SegmentationStrategy | None:
            """Define the segmentation strategy based on a strategy name.

            If the strategy name does not exist, the segmenter instance will not be created.
            It is on the caller to check the type of the return value.

            Throw SegmentationStrategyException on issues with the entities.
            """
            strategy = SegStrategy[name.upper()]
            segmenter = None
            if strategy == SegStrategy.ENTITY:
                if len(metadata.dataset_attrs.entities) == 1:
                    segmenter = strategy.value(
                        X,
                        metadata,
                        dtypes,
                        entity_col=metadata.dataset_attrs.entities[0],
                    )
                else:
                    raise SegmentationStrategyException(
                        "Entity strategy requires only one column"
                    )
            elif strategy == SegStrategy.NN:
                if len(metadata.dataset_attrs.entities) > 0:
                    segmenter = strategy.value(
                        X,
                        metadata,
                        dtypes,
                        columns=metadata.dataset_attrs.entities[0],
                    )
                else:
                    raise SegmentationStrategyException(
                        """
                    Nearest Neighbors strategy requires at least one entities. If you want to specify different columns,
                    specify the list of columns as value for 'segment_by' or use directly 'NNSegmentation' object.
                    """
                    )
            else:
                segmenter = strategy.value(X, metadata, dtypes)
            return segmenter

        segmenter = None

        # The value 'auto takes precedence on everything. None is an alias for 'auto'.
        if strategy is None or strategy == "auto":
            segmenter = auto_strategy(X, metadata, dtypes)

        # `False` always deactivate the segmentation.
        # `True` select the strategy depending on the whether there are entities columns and how many. If none, then it is a fallback for `auto`.
        elif isinstance(strategy, bool):
            if strategy:
                if len(metadata.dataset_attrs.entities) > 0:
                    if len(metadata.dataset_attrs.entities) == 1:
                        segmenter = EntitySegmentation(
                            X,
                            metadata,
                            dtypes,
                            entity_col=metadata.dataset_attrs.entities[0],
                        )
                    else:
                        segmenter = NNSegmentation(
                            X,
                            metadata,
                            dtypes,
                            columns=metadata.dataset_attrs.entities,
                        )
                else:
                    segmenter = auto_strategy(X, metadata, dtypes)
            else:
                segmenter = NoSegmentation(X, metadata, dtypes)

        # Column name takes precedence over the strategy name in the corner case where a column has the same name as a strategy.
        elif isinstance(strategy, str):
            if strategy in metadata.columns.keys():
                segmenter = EntitySegmentation(
                    X, metadata, dtypes, entity_col=strategy)
            else:
                segmenter = strategy_name_to_instance(
                    strategy, X, metadata, dtypes)

        # List of strings are always interpreted as column names.
        elif isinstance(strategy, list):
            not_in_columns = [
                c for c in strategy if c not in metadata.columns.keys()]
            if not_in_columns:
                raise SegmentationStrategyException(
                    """
                    The following columns do not exist: {}
                    """.format(
                        ",".join(not_in_columns)
                    )
                )
            segmenter = NNSegmentation(X, metadata, dtypes, columns=strategy)

        # Manually defined strategy for power-users or internal tests
        elif isinstance(strategy, SegmentationStrategy):
            segmenter = strategy

        if segmenter is None:
            raise SegmentationStrategyException(
                """
                Could not determine any segmentation strategy for the parameter: {}
                        """.format(
                    strategy
                )
            )

        return segmenter

    @staticmethod
    @log_time_factory(logger)
    def _select_slicing_strategy(
        X: Dataset, metadata: Metadata, dtypes: dict[str, Column]
    ) -> SlicingStrategy:
        """Select the slicing strategy to use
        Args:
            X (Dataset): dataset to be segmented
            metadata (Metadata): Metadata corresponding to the dataset
            dtypes: (dict[str, Column]) Dataset schema

        Returns:
            SlicingStrategy: Slicing strategy
        """
        skewed_ratio = Gaussianization.skewness_ratio(metadata)

        if X.shape[1] < 50 or skewed_ratio > Gaussianization.skewed_threshold or metadata._is_multitable:
            return NoSlicing(X, metadata, dtypes)
        return SimpleSlicing(X, metadata, dtypes)

    @log_time_factory(logger)
    def save(self, path: str, copy=False):
        """
        Save the trained `RegularSynthesizer` model to disk.

        This method serializes and stores the trained synthesizer, allowing users
        to reload and reuse it later without retraining. The saved model includes
        all learned patterns, privacy constraints, and configuration settings.

        Args:
            path: The file path where the trained model should be saved. The file should have a `.pkl` or similar extension for serialization.
            copy: If `True`, a deep copy of the synthesizer is saved instead of the original. Defaults to `False`.
        """
        block_folder_name = "blocks"
        save_dir = PurePath(path).parent
        block_folder = Path(save_dir) / block_folder_name
        block_folder.mkdir(parents=True, exist_ok=True)
        for segment_name, segment in self.pipelines.items():
            for slice_name, block_path in segment.blocks.items():
                block_file = PurePath(block_path).name
                dest_path = Path(save_dir) / block_folder_name / block_file
                if copy:
                    shcopy(block_path, dest_path)
                else:
                    if not dest_path.exists():
                        block_path.rename(dest_path)
                self.pipelines[segment_name].blocks[slice_name] = dest_path

        with open(path, "wb") as dump_file:
            dump(self, dump_file)

    @staticmethod
    @log_time_factory(logger)
    def load(path):
        """Load a previously trained `RegularSynthesizer` from disk.

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

        """
        with open(path, 'rb') as dump_file:
            model = load(dump_file)
        return model

    def _check_is_fitted(self) -> bool:
        "Returns a boolean indicating if the synthesizer has been fit or not."
        if hasattr(self, "is_fitted_"):
            if self.is_fitted_:
                return True
        raise NotFittedException(
            "The synthesizer has not yet learned from a given dataset. No samples can be \
generated, please fit the synthesizer first."
        )

    @log_time_factory(logger)
    def _validate_input_values(self, dataset: Dataset,
                                     metadata: Metadata,
                                     condition_on: str | list,
                                     segment_by,
                                     calculated_features: list):

        assert isinstance(
            dataset, Dataset), "X must be an instance of type Dataset. Please provide a valid input."

        if not all(e in metadata.columns for e in list(dataset.columns)):
            raise SynthesizerAssertionError("The provided Metadata does not align with the input Dataset — some columns do not match."
                                            "Please ensure the Metadata corresponds to the correct Dataset.")

        if condition_on is not None:
            if not all(e in list(dataset.columns) for e in condition_on):
                raise SynthesizerAssertionError("Please specify valid columns to condition the synthesis process. "
                                                "Ensure that all columns exist in the provided Dataset.")

        if segment_by is not None:
            print('Validate the segmentation strategy.')

        if calculated_features:
            _validate_calculated_features(calculated_features, dataset)

    @log_time_factory(logger)
    def _anonymizer_post_process(self, output_schema: dict, result: pdDataFrame) -> tuple[dict, pdDataFrame]:
        """Anonymize non synthesized columns, e.g. IDs (unique +
        characteristic.ID) and constant columns.

        1. Columns that have been anonymized might have changed their output dtype
        2. Unique features have to be generated using the anonymizer

        Args:
            output_schema (dict): Mapping of the column to vartype
            result (pdDataFrame): Synthetic sample

        Returns:
            dict, pdDataFrame: output_schema, sample
        """
        # Handle columns with unique values
        unique_features_cols = self._unique_features.keys()
        config = {
            k: v for k, v in self.anonymize.get_config().items()
            if any(
                col for col in v.cols
                if col in unique_features_cols
            )
        }
        builder = AnonymizerConfigurationBuilder(config)

        for v in builder.get_config().values():
            result[v.cols] = v.get_anonymizer()(
                result[v.cols], {e: result.shape[0] for e in v.cols}, **v.params)

        # Handle columns with constant value
        constant_cols = self._constant_features.keys()
        config = {
            k: v for k, v in self.anonymize.get_config().items()
            if any(
                col for col in v.cols
                if col in constant_cols
            )
        }
        builder = AnonymizerConfigurationBuilder(config)

        for v in builder.get_config().values():
            result[v.cols] = v.get_anonymizer()(
                result[v.cols], {e: 1 for e in v.cols}, **v.params)

        for e in self.anonymize.get_config().values():
            for c in e.cols:
                if c in result.columns:
                    if e.type == AnonymizerType.INT:
                        result[c] = result[c].astype("int")
                        output_schema[c] = 'int'
                    else:
                        result[c] = result[c].astype("string")
                        output_schema[c] = 'string'
        return output_schema, result

    @log_time_factory(logger)
    def _calculated_features_post_process(self, output_schema: dict, result: pdDataFrame) -> tuple[dict, pdDataFrame]:
        cf_features = list(
            set([f for cf in self.calculated_features for f in cf.features]))
        if cf_features:
            result = self._apply_calculated_features(result)
            dtypes, _ = humanize_dtypes(dict(result[cf_features].dtypes))
            output_schema = output_schema.copy()
            # The calculated features will appear at the end of the dataset.
            output_schema.update(dtypes)
        return output_schema, result[list(output_schema.keys())]

    @log_time_factory(logger)
    def _write_sample_to_connector(
        self,
        dataset: Dataset,
        connector: BigQueryConnector | ObjectStorageConnector | RDBMSConnector | DatabricksLakehouse | None = None,
        **kwargs
    ):
        logger.info(
            f'Writing sample into connector {connector} with info {kwargs}')
        cwargs = {}
        if isinstance(connector, ObjectStorageConnector):
            cwargs['path'] = kwargs.get('write_path', None)

            write_file_type = kwargs.get('write_file_type', None)
            if write_file_type:
                cwargs['file_type'] = write_file_type

            sep = kwargs.get('write_sep', None)
            if sep:
                cwargs['sep'] = sep

            logger.info(f'writing into object storage with arguments {cwargs}')

            connector.write_file(dataset, **cwargs)
        elif isinstance(connector, RDBMSConnector):
            cwargs['name'] = kwargs.get('write_table', None)

            write_if_exists = kwargs.get('write_if_exists', None)
            if write_if_exists:
                cwargs['if_exists'] = write_if_exists

            logger.info(f'writing into rdbms with arguments {cwargs}')

            connector.write_table(dataset, **cwargs)
        elif isinstance(connector, DatabricksLakehouse):
            cwargs['table'] = kwargs.get('write_table', None)
            cwargs['staging_path'] = kwargs.get('write_staging_path', None)
            cwargs['warehouse'] = kwargs.get('write_warehouse', None)

            write_if_exists = kwargs.get('write_if_exists', None)
            if write_if_exists:
                cwargs['if_exists'] = write_if_exists

            logger.info(f'writing into databricks with arguments {cwargs}')

            connector.write_table(dataset, **cwargs)
