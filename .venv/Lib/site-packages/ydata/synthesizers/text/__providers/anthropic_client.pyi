from _typeshed import Incomplete
from anthropic import Anthropic
from enum import Enum
from functools import cached_property as cached_property
from typing import Any
from ydata.synthesizers.text.__providers.base import BaseLLMClient

logger: Incomplete

class AnthropicModel(str, Enum):
    CLAUDE_3_OPUS = 'claude-3-opus-latest'
    CLAUDE_3_SONNET = 'claude-3-7-sonnet-latest'
    CLAUDE_3_HAIKU = 'claude-3-haiku-latest'

def estimate_batch_anthropic_cost(prompts: Incomplete | None = None, model=..., request_type: str = 'medium', response_type: str = 'medium', output_prompts: Incomplete | None = None, n_prompts: int | None = None): ...

class AnthropicClient(BaseLLMClient):
    api_key: Incomplete
    model: Incomplete
    executor: Incomplete
    def __init__(self, api_key: str, model: AnthropicModel = ..., system_prompt: str | None = None, chat_prompt_template: str | None = None, max_workers: int = 4) -> None: ...
    @cached_property
    def client(self) -> Anthropic: ...
    @cached_property
    def retry_wrapper(self): ...
    def get_max_context_length(self, max_new_tokens: int = 0) -> int: ...
    def generate_batch(self, prompts: list[str], max_tokens: int | None = 4096, temperature: float = 0.7, **kwargs: Any) -> list[str]: ...
