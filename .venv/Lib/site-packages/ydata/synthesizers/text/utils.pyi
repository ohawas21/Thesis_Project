from enum import Enum as Enum, EnumMeta
from typing import Any
from ydata.synthesizers.text.__providers.base import BaseLLMClient

CLIENT_MAP: dict[BaseLLMClient, type]

class CaseInsensitiveEnumMeta(EnumMeta):
    '''
    A metaclass for creating case-insensitive enums.
    This allows enum lookup regardless of case (e.g., "GPT", "gpt", "Gpt" all work).
    '''
    def __getitem__(cls, name: str) -> Any: ...
    def __call__(cls, value: str, *args: Any, **kwargs: Any) -> Any: ...
