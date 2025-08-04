from enum import Enum
from typing import Literal, Type


def enum_to_literal(e: Type[Enum]) -> Literal:
    """Transform an Enum into a Literal for type hinting."""
    return Literal[tuple(e.__members__.keys())]


class EnumToLiteralMixIn(Enum):
    """Mixin to automatically add a `to_literal` function to help with type
    hinting."""
    @classmethod
    def to_literal(cls) -> Literal:
        return enum_to_literal(cls)
