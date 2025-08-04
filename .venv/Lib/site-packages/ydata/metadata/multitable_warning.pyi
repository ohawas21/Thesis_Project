from enum import Enum
from ydata.metadata.warning_types import Level

class MultiTableWarningType(Enum):
    CIRCULAR_REFERENCE = 'circular_reference'
    SELF_REFERENCE = 'self_reference'
    INDEPENDENT_TABLE = 'independent_table'
    NO_RELATIONSHIPS = 'no_relationships'
    INDIRECT_RELATIONSHIP = 'indirect_relationship'

class MultitableWarning:
    """Multitable Warning."""
    def __init__(self, warning_type: MultiTableWarningType, level: Level, tables: list[list[str]], description: str) -> None: ...
    @property
    def type(self) -> MultiTableWarningType:
        """Multitable warning type."""
    @property
    def tables(self) -> list[list[str]]:
        """Tables that the warning apply to."""
    @property
    def level(self) -> Level:
        """Warning level."""
    @property
    def description(self) -> str:
        """Warning description."""

class CircularReferenceWarning(MultitableWarning):
    def __init__(self, tables: list[list[str]]) -> None: ...

class SelfReferenceWarning(MultitableWarning):
    def __init__(self, tables: list[list[str]]) -> None: ...

class IndependentTableWarning(MultitableWarning):
    def __init__(self, tables: list[list[str]]) -> None: ...

class NoRelationshipsWarning(MultitableWarning):
    def __init__(self, tables: list[list[str]]) -> None: ...

class IndirectRelationshipsWarning(MultitableWarning):
    def __init__(self, tables: list[list[str]]) -> None: ...
