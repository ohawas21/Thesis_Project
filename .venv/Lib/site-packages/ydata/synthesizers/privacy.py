from dataclasses import dataclass

import numpy as np
from numpy.random import default_rng
from pandas import DataFrame as pdDataFrame

from ydata.datascience.common import PrivacyLevel
from ydata.preprocessors.methods.gaussianization.rbig.uniform import MarginalHistogramUniformization
from ydata.synthesizers.utils.dp_accounting import calibrate_dp_mechanism
from ydata.utils.data_types import CATEGORICAL_DTYPES, VariableType
from ydata.utils.random import RandomSeed


@dataclass
class PrivacyParameters:
    """Differential privacy parameters."""
    epsilon: float
    delta: float


class DifferentialPrivacyLayer:
    """Differential privacy layer."""

    _DEFAULT_PRIVACY_PARAMETERS = {
        PrivacyLevel.HIGH_PRIVACY: PrivacyParameters(epsilon=0.001, delta=0.01),
        PrivacyLevel.BALANCED_PRIVACY_FIDELITY: PrivacyParameters(epsilon=2.0, delta=0.001),
    }

    _DEFAULT_TIME_SERIES_PRIVACY_PARAMETERS = {
        PrivacyLevel.HIGH_PRIVACY: PrivacyParameters(epsilon=0.001, delta=0.01),
        PrivacyLevel.BALANCED_PRIVACY_FIDELITY: PrivacyParameters(epsilon=1.0, delta=0.001),
    }

    def __init__(self, time_series: bool = False, random_state: RandomSeed = None):
        """Initialize the differential privacy layer.

        Args:
            time_series (bool): Whether the privacy layer will be applied to time series data.
        """
        self._time_series = time_series
        self._orders = (
            [1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5] +
            list(range(5, 64)) + [128, 256, 512]
        )
        self.random_state = random_state

    def _compute_noise(self, epsilon: float, delta: float):
        """Compute differential privacy noise based on the epsilon and delta
        parameters.

        Args:
            epsilon (float): Privacy budget.
            delta (float): Probability of a privacy breach.
        """
        return calibrate_dp_mechanism(
            target_epsilon=epsilon,
            target_delta=delta,
            orders=self._orders
        )

    def apply(self, X: pdDataFrame, privacy_level: PrivacyLevel, input_dtypes: dict):
        """Apply the differential privacy layer to a dataset.

        Args:
            X (pdDataFrame): Dataset that will receive the differential privacy noise.
            privacy_level (PrivacyLevel): Privacy level.
            input_dtypes: (dict): Data type of each column.
        """
        rng = default_rng(seed=self.random_state)
        if privacy_level == PrivacyLevel.HIGH_FIDELITY:
            return X

        privacy_params = self._DEFAULT_TIME_SERIES_PRIVACY_PARAMETERS[privacy_level] \
            if self._time_series \
            else self._DEFAULT_PRIVACY_PARAMETERS[privacy_level]

        privacy_noise = self._compute_noise(epsilon=privacy_params.epsilon,
                                            delta=privacy_params.delta)

        # All values need to be float, otherwise the results are incorrect.
        # Inf values are treated as NaNs in order to be ignored.
        X_flt = X.astype(float).replace([np.inf, -np.inf], np.nan)

        # Columns only containing NaNs are ignored.
        nan_cols = X_flt.columns[X_flt.isna().all()]

        X_np = X_flt.drop(columns=nan_cols).to_numpy()

        hist_unif = MarginalHistogramUniformization(
            X=X_np,
            bound_ext=0.1,
            bins=100,
            alpha=1e-10,
            domain_hint=None,
            privacy_noise=privacy_noise
        )

        X_np = hist_unif.inverse(hist_unif.forward(X_np))
        X_t = pdDataFrame(
            X_np, columns=[k for k in X.columns if k not in nan_cols])

        # If a column only has NaNs, the privacy strategy can't be applied.
        for col in X_t.columns:
            if np.isnan(X_t[col]).all():
                X_t[col] = X[col]

        # All integer columns need to be rounded. NaNs are not supported.
        int_columns = [k for k, v in input_dtypes.items()
                       if v.vartype == VariableType.INT]
        for col in int_columns:
            X_t[col] = X_t[col].fillna(
                value=rng.choice(X_t[col].dropna()))
            X_t[col] = X_t[col].astype(int)

        cat_columns = [k for k, v in input_dtypes.items(
        ) if v.datatype in CATEGORICAL_DTYPES]
        for col in X_t.columns:
            # If a column was reduced to a single value, some noise is added to create diversity.
            # This appears to only happen when the dataset is very small.
            if X_t[col].nunique() == 1:
                X_t[col] = X_t[col] + \
                    rng.normal(0.0, 1.0, len(X_t[col]))

            # If a column has values outside of its domain, they are adjusted to be within
            # the expected range (using a random-based strategy).
            if col in cat_columns:
                invalid_idx = np.where(~np.isin(X_t[col], X[col]))[0]
                replacement_vals = rng.choice(
                    X[col], len(invalid_idx), replace=True)
            else:
                col_data = X[col].replace([np.inf, -np.inf], np.nan)
                col_min = np.nanmin(col_data)
                col_max = np.nanmax(col_data)
                invalid_idx = np.where(
                    ~((X_t[col] >= col_min) & (X_t[col] <= col_max)))[0]
                if np.isnan(col_data).any():
                    invalid_idx = invalid_idx[(
                        ~np.isnan(X_t[col][invalid_idx]))]
                replacement_vals = rng.uniform(
                    col_min, col_max, len(invalid_idx))
            X_t[col][invalid_idx] = replacement_vals

        X_t[nan_cols] = np.nan

        return X_t
