from enum import Enum
from typing import Dict, List, Tuple

from numpy import inf
from numpy import triu as np_triu
from pandas import Interval
from scipy.stats import chisquare

from ydata.utils.data_types import DataType


class Level(Enum):
    """Warning levels."""

    MODERATE = 1
    HIGH = 2


class WarningType(Enum):
    """Warning types."""

    SKEWNESS = "skewness"
    MISSINGS = "missings"
    CARDINALITY = "cardinality"
    DUPLICATES = "duplicates"
    IMBALANCE = "imbalance"
    CONSTANT = "constant"
    INFINITY = "infinity"
    ZEROS = "zeros"
    CORRELATION = "correlation"
    UNIQUE = "unique"
    UNIFORM = "uniform"
    CONSTANT_LENGTH = "constant_length"


class WarningOrientation(Enum):
    """Orientation of Warning computation."""

    COLUMN = "column"
    DATASET = "dataset"


WARNING_MAPS = {
    DataType.NUMERICAL: [
        WarningType.SKEWNESS,
        WarningType.MISSINGS,
        WarningType.CONSTANT,
        WarningType.INFINITY,
        WarningType.ZEROS,
        WarningType.UNIFORM,
    ],
    DataType.CATEGORICAL: [
        WarningType.MISSINGS,
        WarningType.CARDINALITY,
        WarningType.CONSTANT,
        WarningType.UNIQUE,
        WarningType.UNIFORM,
        WarningType.CONSTANT_LENGTH,
        WarningType.IMBALANCE,
    ],
    DataType.DATE: [WarningType.MISSINGS, WarningType.CONSTANT, WarningType.UNIQUE],
    DataType.STR: [
        WarningType.MISSINGS,
        WarningType.CARDINALITY,
        WarningType.CONSTANT,
        WarningType.UNIQUE,
        WarningType.UNIFORM,
        WarningType.CONSTANT_LENGTH,
        WarningType.IMBALANCE,
    ],
    DataType.LONGTEXT: [
        WarningType.MISSINGS,
        WarningType.UNIQUE,
        WarningType.CONSTANT_LENGTH,
    ],
}


class Warning:
    """Data warning type."""

    def __init__(self, warning_type: WarningType, details: dict, column: dict = None):
        self.column = column
        self.type = warning_type
        self.details = details

    def _validation(self, value):
        if value == WarningType.DUPLICATES and self.column is not None:
            raise Exception(
                f"The warning of type {WarningType.DUPLICATES.value} "
                + "it's only valid for the all dataset. Column input is not valid."
            )

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        value = self.details["value"]
        f_value = "{:.2f}".format(value) if isinstance(value, float) else value
        return f"Warning(column={self.column}, type={self.type}, level={self.details['level']}, value={f_value})"


class BaseWarning:
    """Base Warning."""

    def __init__(
        self,
        warning_type: WarningType,
        orientation: WarningOrientation,
        metric_name: str = None,
    ) -> None:
        self._type = warning_type
        self._orientation = orientation
        self._intervals = self._create_intervals()
        self._metric_name = metric_name

    def evaluate(self, summary: dict, dtypes: dict) -> List[Warning]:
        """Create warnings based on Metadata summary.

        Args:
            summary (Dict): Metadata summary

        Returns:
            List[Warning]: List of the raised warnings
        """
        if self.orientation == WarningOrientation.DATASET:
            return self._evaluate_dataset(summary, dtypes)

        return self._evaluate_columns(summary, dtypes)

    def _evaluate_columns(self, summary: dict, dtypes: dict) -> List[Warning]:
        metric_name = self.type.name.lower()
        if self._metric_name:
            metric_name = self._metric_name

        metric = summary[metric_name]

        warnings = []
        for col, value in metric.items():
            if self.type not in WARNING_MAPS[dtypes[col]]:
                continue

            is_warning, warning_level = self._get_warning_level(value)

            if is_warning:
                warnings.append(self._create_warning(
                    col, warning_level, value))
        return warnings

    def _evaluate_dataset(self, summary: dict, dtypes: dict) -> List[Warning]:
        metric = summary[self.type.name.lower()]

        is_warning, warning_level = self._get_warning_level(metric)
        if is_warning:
            return [self._create_warning("dataset", warning_level, metric)]
        return []

    def _get_warning_level(self, value) -> Tuple:
        """Calculate the warning level."""
        is_warning = False
        level = None
        if any(map(lambda interval: value in interval, self._intervals[Level.HIGH])):
            is_warning = True
            level = Level.HIGH
        elif any(
            map(lambda interval: value in interval,
                self._intervals[Level.MODERATE])
        ):
            is_warning = True
            level = Level.MODERATE

        return is_warning, level

    def _create_intervals(self) -> Dict[str, List[Interval]]:
        return {Level.HIGH: [], Level.MODERATE: []}

    def _create_warning(self, column, level, value):
        return Warning(
            warning_type=self.type,
            column=column,
            details={"level": level, "value": value},
        )

    @property
    def type(self):
        """Warning type."""
        return self._type

    @property
    def orientation(self):
        """Warning orientation."""
        return self._orientation


class WarningEngine:
    """Warning Engine."""

    warnings: Dict[str, BaseWarning]

    def __init__(self, warnings: Dict[str, BaseWarning]) -> None:
        self.warnings = warnings

    def evaluate(self, summary: dict, dtypes: dict):
        """Evaluate all Warnings."""
        result = {}

        for name, warning in self.warnings.items():
            result[name] = warning.evaluate(summary, dtypes)
        return result


class SkewnessWarning(BaseWarning):
    """Skweness Warning."""

    def __init__(self):
        """Empty initializer."""
        super().__init__(WarningType.SKEWNESS, WarningOrientation.COLUMN)

    # TODO add config yml intervals
    def _create_intervals(self) -> Dict[str, List[Interval]]:
        return {
            Level.MODERATE: [Interval(-1, -0.5, "both"), Interval(0.5, 1, "both")],
            Level.HIGH: [Interval(-inf, -1, "left"), Interval(1, inf, "right")],
        }


class DuplicatesWarning(BaseWarning):
    """Duplicates Warning."""

    def __init__(self):
        """Empty initializer."""
        super().__init__(WarningType.DUPLICATES, WarningOrientation.DATASET)

    def _create_intervals(self) -> Dict[str, List[Interval]]:
        return {
            Level.MODERATE: [Interval(0.1, 0.3, "right")],
            Level.HIGH: [Interval(0.3, inf, "right")],
        }

    def _evaluate_dataset(self, summary: dict, dtypes: dict) -> List[Warning]:
        metric = summary[self.type.name.lower()]
        if summary['nrows'] == 0:
            return []
        fraction_duplicates = metric / summary['nrows']
        is_warning, warning_level = self._get_warning_level(
            fraction_duplicates)
        if is_warning:
            return [self._create_warning("dataset", warning_level, fraction_duplicates)]
        return []


class HighCardinalityWarning(BaseWarning):
    """High Cardinality Warning."""

    def __init__(self):
        """Empty initializer."""
        super().__init__(WarningType.CARDINALITY, WarningOrientation.COLUMN)

    def _create_intervals(self) -> Dict[str, List[Interval]]:
        return {
            Level.MODERATE: [Interval(20, 50, "neither")],
            Level.HIGH: [Interval(50, inf, "both")],
        }


class ImbalanceWarning(BaseWarning):
    """High Cardinality Warning."""

    def __init__(self):
        """Empty initializer."""
        super().__init__(WarningType.IMBALANCE, WarningOrientation.COLUMN)

    def _create_intervals(self) -> Dict[str, List[Interval]]:
        return {
            Level.HIGH: [Interval(0.5, 1, "right")],
            Level.MODERATE: [Interval(0.15, 0.5, "both")],
        }


class MissingValuesWarning(BaseWarning):
    """Missing Values Warning."""

    def __init__(self):
        """Empty initializer."""
        super().__init__(WarningType.MISSINGS, WarningOrientation.COLUMN)

    def _create_intervals(self) -> Dict[str, List[Interval]]:
        return {
            Level.MODERATE: [Interval(0.3, 0.6, "right")],
            Level.HIGH: [Interval(0.6, inf, "right")],
        }


class ConstantWarning(BaseWarning):
    """Constant Warning."""

    def __init__(self):
        """Empty initializer."""
        super().__init__(WarningType.CONSTANT, WarningOrientation.COLUMN, "cardinality")

    def _create_intervals(self) -> Dict[str, List[Interval]]:
        return {
            Level.MODERATE: [],
            Level.HIGH: [Interval(1, 1, "both")],
        }

    def _evaluate_columns(self, summary: dict, dtypes: dict) -> List[Warning]:
        cardinality = summary[self._metric_name]
        missings = summary['missings']
        nrows = summary["nrows"]

        warnings = []
        for col in list(dtypes.keys()):
            if self.type not in WARNING_MAPS[dtypes[col]]:
                continue
            n_miss = missings[col]
            if (nrows - n_miss) == 0:
                value = 1
            elif (cardinality[col]==1 and n_miss==0):
                value = 1
            else:
                value = 0
            is_warning, warning_level = self._get_warning_level(value)

            if is_warning:
                warnings.append(self._create_warning(
                    col, warning_level, value))
        return warnings


class ZerosWarning(BaseWarning):
    """Zeros Warning."""

    def __init__(self):
        """Empty initializer."""
        super().__init__(WarningType.ZEROS, WarningOrientation.COLUMN)

    def _create_intervals(self) -> Dict[str, List[Interval]]:
        return {
            Level.MODERATE: [Interval(0.01, 0.25, "right")],
            Level.HIGH: [Interval(0.25, inf, "right")],
        }


class InfinityWarning(BaseWarning):
    """Infinity Warning."""

    def __init__(self):
        """Empty initializer."""
        super().__init__(WarningType.INFINITY, WarningOrientation.COLUMN)

    def _create_intervals(self) -> Dict[str, List[Interval]]:
        return {
            Level.MODERATE: [],
            Level.HIGH: [Interval(1, inf, "right")],
        }


class CorrelationWarning(BaseWarning):
    """Correlation Warning."""

    def __init__(self):
        """Empty initializer."""
        super().__init__(WarningType.CORRELATION, WarningOrientation.COLUMN)

    def _create_intervals(self) -> Dict[str, List[Interval]]:
        return {
            # TODO define this intervals
            Level.MODERATE: [
                Interval(-0.7, -0.45, "left"),
                Interval(0.45, 0.7, "right"),
            ],
            Level.HIGH: [Interval(-inf, -0.7, "left"), Interval(0.7, inf, "right")],
        }

    def _evaluate_columns(self, summary: dict, dtypes: dict) -> List[Warning]:
        metric = summary[self.type.name.lower()]
        warnings = []
        metric = metric.where(np_triu(metric, 1).astype(bool)).stack()
        metric = metric.rename_axis(
            ("col1", "col2")).reset_index(name="correlation")
        for item in metric.to_dict(orient="records"):
            value = item["correlation"]
            is_warning, warning_level = self._get_warning_level(value)

            if is_warning:
                warnings.append(
                    self._create_warning(
                        item["col1"] + "|" + item["col2"], warning_level, value
                    )
                )
        return warnings


class UniqueWarning(BaseWarning):
    """Unique Warning."""

    def __init__(self):
        """Empty initializer."""
        super().__init__(WarningType.UNIQUE, WarningOrientation.COLUMN, "cardinality")

    def _create_intervals(self) -> Dict[str, List[Interval]]:
        return {
            Level.MODERATE: [Interval(0.9, 1, "left")],
            Level.HIGH: [Interval(1, inf, "both")],
        }

    def _evaluate_columns(self, summary: dict, dtypes: dict) -> List[Warning]:
        metric = summary[self._metric_name]
        warnings = []
        nrows = summary["nrows"]
        missings = summary["missings"]
        for col, unique in metric.items():
            if self.type not in WARNING_MAPS[dtypes[col]]:
                continue
            miss = missings[col]
            if (nrows - miss) == 0:
                continue
            value = unique / (nrows - miss)
            is_warning, warning_level = self._get_warning_level(value)

            if is_warning:
                warnings.append(self._create_warning(
                    col, warning_level, value))
        return warnings


class UniformWarning(BaseWarning):
    """Uniform Warning."""

    def __init__(self):
        """Empty initializer."""
        super().__init__(WarningType.UNIFORM, WarningOrientation.COLUMN, "value_counts")

    def _create_intervals(self) -> Dict[str, List[Interval]]:
        return {
            Level.MODERATE: [Interval(0.9, 0.999, "left")],
            Level.HIGH: [Interval(0.999, inf, "left")],
        }

    def _evaluate_columns(self, summary: dict, dtypes: dict) -> List[Warning]:
        metric = summary[self._metric_name]
        warnings = []
        for col, unique in metric.items():
            if WarningType.UNIFORM not in WARNING_MAPS[dtypes[col]]:
                continue
            value = chisquare(unique).pvalue
            is_warning, warning_level = self._get_warning_level(value)

            if is_warning:
                warnings.append(self._create_warning(
                    col, warning_level, value))
        return warnings


class ConstantLengthWarning(BaseWarning):
    """Constant length Warning."""

    def __init__(self):
        """Empty initializer."""
        super().__init__(
            WarningType.CONSTANT_LENGTH, WarningOrientation.COLUMN, "string_len"
        )

    def _create_intervals(self) -> Dict[str, List[Interval]]:
        return {
            Level.MODERATE: [],
            Level.HIGH: [Interval(0, 0, "both")],
        }

    def _evaluate_columns(self, summary: dict, dtypes: dict) -> List[Warning]:
        metric = summary[self._metric_name]
        warnings = []
        for col, lengths in metric.items():
            if self.type not in WARNING_MAPS[dtypes[col]]:
                continue
            value = lengths["max"] - lengths["min"]
            is_warning, warning_level = self._get_warning_level(value)

            if is_warning:
                warnings.append(self._create_warning(
                    col, warning_level, lengths["max"]))
        return warnings
