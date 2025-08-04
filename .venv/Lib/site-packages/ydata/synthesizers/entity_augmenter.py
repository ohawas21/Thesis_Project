from dataclasses import dataclass
from typing import List, Literal, Optional, Union

import pandas as pd
from numpy import array as nparray
from numpy.random import default_rng
from pandas import DataFrame as pdDataFrame
from scipy.signal import savgol_filter

from ydata.metadata import Metadata
from ydata.preprocessors.methods.gaussianization.rbig.config import RBIGConfig
from ydata.preprocessors.methods.gaussianization.rbig.model import RBIG
from ydata.utils.random import RandomSeed


@dataclass
class SmoothingConfig:
    """Smoothing Configuration.

    enabled (bool): enable or not the smoothing. window (float): time
    window to use for smoothing. Expressed as a fraction of the original
    dataset size. degree (int): degree of the polynomial to use for
    smoothing. derivative (int): order of the derivative to consider for
    smoothing
    """
    enabled: bool = False
    window: float = 0.0003
    degree: int = 2
    derivative: int = 0


@dataclass
class FidelityConfig:
    """Fidelity Configuration.

    strategy (str): strategy to use for the behavioral noise. noise
    (float): noise amplitude expressed as a fraction of the standard
    deviation. Can theoreically be higher than 1.
    """
    strategy: Literal['gaussian', 'uniform'] = 'gaussian'
    noise: float = 0.3


# @typechecked
class EntityAugmenter:

    __INDEX_ENTITIES = '__index_entities'

    def __init__(self, X: pdDataFrame, metadata: Metadata, n_entities: int, pivot_columns: List[str]):
        self.nrows: int = int(X.shape[0])
        self.n_entities: int = n_entities
        self.original_sbk = {}
        self.pert_blocks = {}
        self.gauss_models = {}
        self.pivot_columns: List[str] = pivot_columns
        self.all_columns:  List[str] = list(metadata.columns.keys())
        self.sortbykey: List[str] = metadata.dataset_attrs.sortbykey
        self.entities: List[str] = metadata.dataset_attrs.entities
        self.smoothing_strategy = None
        self.smoother_per_column = False
        self.__default_cfg_key = '__default'
        self.fidelity_strategy = None
        self.fidelity_per_column = False
        self.gauss_config = RBIGConfig(
            uniformizer="hist",
            rotation="PCA",
            max_iter=100,
            zero_tolerance=10,
            bins=100,
            alpha=1e-10,
            eps=1e-10,
            max_layers=1_000,
            domain_hint=None)

    def _init_smoothing_strategy(self, smoothing: Union[bool, dict, SmoothingConfig]):
        """Initialize the smoothing strategy based on a given configuration."""
        smoothing_cfg = smoothing
        if isinstance(smoothing, bool):
            smoothing_cfg = SmoothingConfig(enabled=smoothing)
        elif isinstance(smoothing, dict):
            try:
                smoothing_cfg = SmoothingConfig(**smoothing)
            except Exception:
                for c, cfg in smoothing.items():
                    if c in self.all_columns:
                        smoothing_cfg[c] = SmoothingConfig(**cfg)
                smoothing_cfg[self.__default_cfg_key] = SmoothingConfig(
                    **{k: v for k, v in smoothing.items() if k not in self.all_columns})
                self.smoother_per_column = True
        return smoothing_cfg

    def _init_fidelity_strategy(self, fidelity: Optional[Union[float, dict, FidelityConfig]]):
        """Initialize the fidelity strategy based on a given configuration."""
        fidelity_cfg = fidelity
        if fidelity is None:
            fidelity_cfg = FidelityConfig()
        elif isinstance(fidelity, float):
            fidelity_cfg = FidelityConfig(noise=fidelity)
        elif isinstance(fidelity, dict):
            try:
                fidelity_cfg = FidelityConfig(**fidelity)
            except Exception:
                for c, cfg in fidelity.items():
                    if c in self.all_columns:
                        fidelity_cfg[c] = FidelityConfig(**cfg)
                fidelity_cfg[self.__default_cfg_key] = FidelityConfig(
                    **{k: v for k, v in fidelity.items() if k not in self.all_columns})
                self.fidelity_per_column = True
        return fidelity_cfg

    def _sample_new_entities(self,
                             block_name: str,
                             block_synth,
                             block_preprocessor,
                             smoothing: Union[bool, dict,
                                              SmoothingConfig] = False,
                             fidelity: Optional[Union[float,
                                                      dict, FidelityConfig]] = None,
                             bootstrapping: Optional[pdDataFrame] = None,
                             random_state: RandomSeed = None):
        """Generate new entities based on a given block synthesizer and
        preprocessor and a smoothing and fidelity strategy,

        The augmentation works in two steps:
            1. Pivot columns are generated by adding behavioral noise to the distribution of differences that has been gaussianized during the training process
            2. The pivot columns are passed to the synthesizer as `bootstrapping` or `conditionning` columns. The synthesizer generates the other columns based on the pivot columns.

        This procedure will always generate as many entities as in the orginal dataset. Not less, not more. This restriction is due to the gaussianization process limitations on the
        input/output dimensions.
        """
        self.random_state = random_state

        self.smoothing_strategy = self._init_smoothing_strategy(smoothing)
        self.fidelity_strategy = self._init_fidelity_strategy(fidelity)

        pivot_df = self._get_entity_pivot(block_name)
        if bootstrapping is not None:
            pivot_df = pivot_df.drop(columns=bootstrapping.columns)
            pivot_df = pd.concat([bootstrapping, pivot_df], axis=1)

        new_trajectories_raw = block_synth.sample(
            n_samples=pivot_df.shape[0], bootstrapping=pivot_df)
        new_trajectories_raw = new_trajectories_raw.sort_values(
            by=self.sortbykey + self.entities, ignore_index=True)

        new_trajectories = block_preprocessor.inverse_transform(
            new_trajectories_raw)
        return new_trajectories

    def _get_entity_pivot(self, block_name: str):
        """Get entity pivot columns.

        The pivot columns are obtained by adding noise and smoothing to
        the output of the `fit_block_bootstraper`.
        """
        rng = default_rng(seed=self.random_state)
        df_step_1_pert = self.pert_blocks[block_name].copy()

        def generate_noise(self, df, c, fidelity_strategy: FidelityConfig):
            # Right now, the concrete strategy is defined in this if/else.
            # If this grows, it might be better to outsource it to a specific auxiliary class/operator.
            noise = fidelity_strategy.noise
            if fidelity_strategy.strategy == 'uniform':
                noise_df = noise * df[c].std() * \
                    rng.choice([-1, 0, 1], df[c].shape[0])
            else:
                noise_df = rng.normal(
                    0,
                    noise * df[c].std(),
                    df.shape[0]
                )
            return noise_df

        for c in self.pivot_columns:
            if self.fidelity_per_column:
                default_strategy = self.fidelity_strategy[self.__default_cfg_key]
                strategy = self.fidelity_strategy.get(c, default_strategy)
            else:
                strategy = self.fidelity_strategy

            df_step_1_pert[c] = df_step_1_pert[c] + generate_noise(
                self, df_step_1_pert, c, strategy)

        X_back = self.gauss_models[block_name].inverse_transform(
            df_step_1_pert[[f'{EntityAugmenter.__INDEX_ENTITIES}_G'] + self.pivot_columns].to_numpy())
        df_back = pdDataFrame(
            X_back, columns=[f'{EntityAugmenter.__INDEX_ENTITIES}_Gb'] + self.pivot_columns)

        df_back[self.entities] = df_step_1_pert[self.entities]

        df_back_smooth = df_back.copy()
        for c in self.pivot_columns:
            enabled = self.smoothing_strategy.enabled if not self.smoother_per_column else self.smoothing_strategy.get(
                c, self.__default_cfg_key).enabled
            if enabled:
                polyorder = self.smoothing_strategy.degree if not self.smoother_per_column else self.smoothing_strategy.get(
                    c, self.__default_cfg_key).degree
                window = self.smoothing_strategy.window if not self.smoother_per_column else self.smoothing_strategy.get(
                    c, self.__default_cfg_key).window
                derivative = self.smoothing_strategy.derivative if not self.smoother_per_column else self.smoothing_strategy.get(
                    c, self.__default_cfg_key).derivative
                window_length = max(
                    polyorder + 1, 2 * int(df_back_smooth[c].shape[0] * window / 2) + 1)
                df_back_smooth[c] = savgol_filter(
                    df_back_smooth[c], window_length=window_length, polyorder=polyorder, deriv=derivative)

        df_back_smooth[self.original_sbk[block_name].columns] = self.original_sbk[block_name]
        return df_back_smooth[self.sortbykey + self.entities + self.pivot_columns]

    def fit_block_bootstraper(self, block_name: str, X: pdDataFrame):
        """The entity augmenter works by bootstrapping the synthesizer with few
        columns called `pivot` obtained in few steps:

        1. Gaussianization of the original columns order in by sortbykey + entities - by this order we preserve all dependencies
        2. Sorting by entities + sortbykey
        3. Calculating the distribution of difference per entity
        4. Sort back to the exact same order to avoid issues with previous preprocessing steps done outside the entity augmenter
        """
        self.gauss_models[block_name] = RBIG(self.gauss_config)

        X = X.rename_axis(
            EntityAugmenter.__INDEX_ENTITIES).reset_index(drop=False)

        self.original_sbk[block_name] = X.sort_values(
            by=self.entities + self.sortbykey)[self.sortbykey].reset_index(drop=True).copy()
        X_df_train = X.sort_values(
            by=self.sortbykey + self.entities, ignore_index=True)
        X_df_train = X[[EntityAugmenter.__INDEX_ENTITIES] + self.pivot_columns]
        X_df_train = X_df_train.fillna(method='ffill').fillna(0)

        self.gauss_models[block_name].fit(
            nparray(X_df_train), progress_bar=False)

        X_step_1 = self.gauss_models[block_name].transform(
            X_df_train[[EntityAugmenter.__INDEX_ENTITIES] + self.pivot_columns].to_numpy())
        df_step_1 = pdDataFrame(X_step_1, columns=[
                                f'{EntityAugmenter.__INDEX_ENTITIES}_G'] + self.pivot_columns)

        df_step_1[EntityAugmenter.__INDEX_ENTITIES] = X[EntityAugmenter.__INDEX_ENTITIES]
        df_step_1[self.entities] = X[self.entities]

        self.pert_blocks[block_name] = df_step_1.sort_values(
            by=self.entities + [EntityAugmenter.__INDEX_ENTITIES], ignore_index=True)
