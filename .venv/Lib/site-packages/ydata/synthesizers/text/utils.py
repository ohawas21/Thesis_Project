"""
    Utility functions for the text synthetic data generator model.
"""
from pathlib import Path
from typing import Union, Tuple, Optional, Type, Any

from enum import Enum, EnumMeta

import pyarrow as pa


from ydata.synthesizers.text.__providers.base import (LLMProvider, BaseLLMClient)
from ydata.synthesizers.text.__providers import (AnthropicClient,
                                                 OpenAIClient)

# Provider-to-client mapping
CLIENT_MAP: dict[BaseLLMClient, Type] = {
    LLMProvider.OPENAI: OpenAIClient,
    LLMProvider.ANTHROPIC: AnthropicClient,
}

class CaseInsensitiveEnumMeta(EnumMeta):
    """
    A metaclass for creating case-insensitive enums.
    This allows enum lookup regardless of case (e.g., "GPT", "gpt", "Gpt" all work).
    """
    def __getitem__(cls, name: str) -> Any:
        try:
            return super().__getitem__(name)
        except KeyError:
            # Try case-insensitive lookup
            try:
                return cls._member_map_[name.lower()]
            except KeyError:
                raise KeyError(f"'{name}' is not a valid {cls.__name__}")

    def __call__(cls, value: str, *args: Any, **kwargs: Any) -> Any:
        if isinstance(value, str):
            try:
                # Try exact match first
                return super().__call__(value, *args, **kwargs)
            except ValueError:
                # Try case-insensitive match
                try:
                    return cls._value2member_map_[value.lower()]
                except KeyError:
                    raise ValueError(f"'{value}' is not a valid {cls.__name__}")
        return super().__call__(value, *args, **kwargs)

# TODO needs improvements - remove return values and raise errors
# TODO: validate logic when the STR is a path to a file and not a directory
def _validate_input_path(input_source: Union[str, pa.Table]) -> Tuple[bool, Optional[Path], str]:
    """
    Validate the input path and determine if it's a file or directory.

    Args:
        input_source: Either a path to a document/folder or a pyarrow Table

    Returns:
        Tuple[bool, Optional[Path], str]: A tuple containing:
            - bool: True if input is a valid path, False if it's a Table
            - Optional[Path]: The validated Path object if input is a string, None otherwise
            - str: The type of input ("file", "directory", or "table")

    Raises:
        ValueError: If input_source is a string but not a valid file or directory
        ValueError: If input_source is neither a string nor a pyarrow Table
    """
    if isinstance(input_source, str):
        input_path = Path(input_source)
        if not input_path.exists():
            raise ValueError(f"Input path does not exist: {input_source}")
        if not (input_path.is_file() or input_path.is_dir()):
            raise ValueError(f"Input path must be a file or directory: {input_source}")

        if input_path.is_dir():
            return False, None
        else:
            return True, input_path.suffix.lower()
