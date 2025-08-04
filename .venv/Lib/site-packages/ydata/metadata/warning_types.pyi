from _typeshed import Incomplete
from enum import Enum

class Level(Enum):
    """Warning levels."""
    MODERATE = 1
    HIGH = 2

class WarningType(Enum):
    """Warning types."""
    SKEWNESS = 'skewness'
    MISSINGS = 'missings'
    CARDINALITY = 'cardinality'
    DUPLICATES = 'duplicates'
    IMBALANCE = 'imbalance'
    CONSTANT = 'constant'
    INFINITY = 'infinity'
    ZEROS = 'zeros'
    CORRELATION = 'correlation'
    UNIQUE = 'unique'
    UNIFORM = 'uniform'
    CONSTANT_LENGTH = 'constant_length'

class WarningOrientation(Enum):
    """Orientation of Warning computation."""
    COLUMN = 'column'
    DATASET = 'dataset'

WARNING_MAPS: Incomplete

class Warning:
    """Data warning type."""
    column: Incomplete
    type: Incomplete
    details: Incomplete
    def __init__(self, warning_type: WarningType, details: dict, column: dict = None) -> None: ...

class BaseWarning:
    """Base Warning."""
    def __init__(self, warning_type: WarningType, orientation: WarningOrientation, metric_name: str = None) -> None: ...
    def evaluate(self, summary: dict, dtypes: dict) -> list[Warning]:
        """Create warnings based on Metadata summary.

        Args:
            summary (Dict): Metadata summary

        Returns:
            List[Warning]: List of the raised warnings
        """
    @property
    def type(self):
        """Warning type."""
    @property
    def orientation(self):
        """Warning orientation."""

class WarningEngine:
    """Warning Engine."""
    warnings: dict[str, BaseWarning]
    def __init__(self, warnings: dict[str, BaseWarning]) -> None: ...
    def evaluate(self, summary: dict, dtypes: dict):
        """Evaluate all Warnings."""

class SkewnessWarning(BaseWarning):
    """Skweness Warning."""
    def __init__(self) -> None:
        """Empty initializer."""

class DuplicatesWarning(BaseWarning):
    """Duplicates Warning."""
    def __init__(self) -> None:
        """Empty initializer."""

class HighCardinalityWarning(BaseWarning):
    """High Cardinality Warning."""
    def __init__(self) -> None:
        """Empty initializer."""

class ImbalanceWarning(BaseWarning):
    """High Cardinality Warning."""
    def __init__(self) -> None:
        """Empty initializer."""

class MissingValuesWarning(BaseWarning):
    """Missing Values Warning."""
    def __init__(self) -> None:
        """Empty initializer."""

class ConstantWarning(BaseWarning):
    """Constant Warning."""
    def __init__(self) -> None:
        """Empty initializer."""

class ZerosWarning(BaseWarning):
    """Zeros Warning."""
    def __init__(self) -> None:
        """Empty initializer."""

class InfinityWarning(BaseWarning):
    """Infinity Warning."""
    def __init__(self) -> None:
        """Empty initializer."""

class CorrelationWarning(BaseWarning):
    """Correlation Warning."""
    def __init__(self) -> None:
        """Empty initializer."""

class UniqueWarning(BaseWarning):
    """Unique Warning."""
    def __init__(self) -> None:
        """Empty initializer."""

class UniformWarning(BaseWarning):
    """Uniform Warning."""
    def __init__(self) -> None:
        """Empty initializer."""

class ConstantLengthWarning(BaseWarning):
    """Constant length Warning."""
    def __init__(self) -> None:
        """Empty initializer."""
