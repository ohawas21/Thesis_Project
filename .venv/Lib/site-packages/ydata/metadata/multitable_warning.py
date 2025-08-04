from enum import Enum

from ydata.metadata.warning_types import Level


class MultiTableWarningType(Enum):
    CIRCULAR_REFERENCE = "circular_reference"
    SELF_REFERENCE = "self_reference"
    INDEPENDENT_TABLE = "independent_table"
    NO_RELATIONSHIPS = "no_relationships"
    INDIRECT_RELATIONSHIP = "indirect_relationship"


class MultitableWarning:
    """Multitable Warning."""

    def __init__(
        self,
        warning_type: MultiTableWarningType,
        level: Level,
        tables: list[list[str]],
        description: str,
    ) -> None:
        self._type = warning_type
        self._level = level
        self._tables = tables
        self._description = description

    @property
    def type(self) -> MultiTableWarningType:
        """Multitable warning type."""
        return self._type

    @property
    def tables(self) -> list[list[str]]:
        """Tables that the warning apply to."""
        return self._tables

    @property
    def level(self) -> Level:
        """Warning level."""
        return self._level

    @property
    def description(self) -> str:
        """Warning description."""
        return self._description

    def __repr__(self) -> str:
        return str(self)


class CircularReferenceWarning(MultitableWarning):
    def __init__(self, tables: list[list[str]]) -> None:
        super().__init__(
            warning_type=MultiTableWarningType.CIRCULAR_REFERENCE,
            tables=tables,
            level=Level.HIGH,
            description="One or more potential circular references found.",
        )

    def __str__(self) -> str:
        return f"CircularReferenceWarning(tables={self.tables}, level={self.level})"


class SelfReferenceWarning(MultitableWarning):
    def __init__(self, tables: list[list[str]]) -> None:
        super().__init__(
            warning_type=MultiTableWarningType.SELF_REFERENCE,
            tables=tables,
            level=Level.MODERATE,
            description=f"Self reference found in {len(tables)} tables.",
        )

    def __str__(self) -> str:
        return f"SelfReferenceWarning(tables={self.tables}, level={self.level})"


class IndependentTableWarning(MultitableWarning):
    def __init__(self, tables: list[list[str]]) -> None:
        super().__init__(
            warning_type=MultiTableWarningType.INDEPENDENT_TABLE,
            tables=tables,
            level=Level.MODERATE,
            description=f"Found {len(tables)} with no relationship with any other table",
        )

    def __str__(self) -> str:
        return f"IndependentTableWarning(tables={self.tables}, level={self.level})"


class NoRelationshipsWarning(MultitableWarning):
    def __init__(self, tables: list[list[str]]) -> None:
        super().__init__(
            warning_type=MultiTableWarningType.NO_RELATIONSHIPS,
            tables=tables,
            level=Level.MODERATE,
            description="No relationship present in the database",
        )

    def __str__(self) -> str:
        return f"NoRelationshipsWarning(tables={self.tables}, level={self.level})"


class IndirectRelationshipsWarning(MultitableWarning):
    def __init__(self, tables: list[list[str]]) -> None:
        super().__init__(
            warning_type=MultiTableWarningType.INDIRECT_RELATIONSHIP,
            tables=tables,
            level=Level.HIGH,
            description="One or more indirect relationships found.",
        )

    def __str__(self) -> str:
        return f"IndirectRelationshipsWarning(tables={self.tables}, level={self.level})"
