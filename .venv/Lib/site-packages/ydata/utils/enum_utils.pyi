from enum import Enum
from typing import Literal

def enum_to_literal(e: type[Enum]) -> Literal:
    """Transform an Enum into a Literal for type hinting."""

class EnumToLiteralMixIn(Enum):
    """Mixin to automatically add a `to_literal` function to help with type
    hinting."""
    @classmethod
    def to_literal(cls) -> Literal: ...
