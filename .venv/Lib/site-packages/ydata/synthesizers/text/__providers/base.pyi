import abc
import pyarrow as pa
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

class LLMProvider(Enum):
    """Enum for supported LLM providers."""
    OPENAI = 'openai'
    ANTHROPIC = 'anthropic'

content_lengths: Incomplete

class BaseLLMClient(ABC, metaclass=abc.ABCMeta):
    """Abstract base class for LLM clients."""
    system_prompt: Incomplete
    chat_prompt_template: Incomplete
    def __init__(self, system_prompt: str | None = None, chat_prompt_template: str | None = None) -> None: ...
    @abstractmethod
    def get_max_context_length(self, max_new_tokens: int) -> int: ...
    def generate(self, table: pa.Table, prompt_column: str = 'prompt', in_context_examples: list[str] | None = None, end_instruction: str | None = None, sep: str = '\n', min_in_context_examples: int | None = None, max_in_context_examples: int | None = None, max_tokens: int | None = None, temperature: float = 0.7, return_table: bool = False, **kwargs: Any) -> list[str] | pa.Table:
        """
        Main user-facing generation method. Handles steps formatting and generation.
        Takes a pyarrow.Table as input and applies batch generation based on a specified column.
        """
    def count_tokens(self, text: str, model_name: str | None = None) -> int: ...
    def unload_model(self) -> None: ...
