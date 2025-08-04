from _typeshed import Incomplete
from enum import Enum
from functools import cached_property as cached_property
from openai import OpenAI
from typing import Any
from ydata.synthesizers.text.__providers.base import BaseLLMClient

logger: Incomplete

class OpenAIModel(str, Enum):
    GPT_3_5_TURBO = 'gpt-3.5-turbo'
    GPT_4 = 'gpt-4'
    GPT_4_TURBO = 'gpt-4-turbo'
    GPT_4o = 'gpt-4o'
    GPT_4o_mini = 'gpt-4o-mini'

def estimate_batch_openai_cost(prompts: Incomplete | None = None, model: str = 'gpt-4-turbo', request_type: str = 'medium', response_type: str = 'medium', output_prompts: Incomplete | None = None, n_prompts: int | None = None): ...

class OpenAIClient(BaseLLMClient):
    api_key: Incomplete
    model: Incomplete
    executor: Incomplete
    def __init__(self, api_key: str, model: OpenAIModel = ..., system_prompt: str | None = None, chat_prompt_template: str | None = None, max_workers: int = 4) -> None: ...
    @cached_property
    def client(self) -> OpenAI: ...
    def get_max_context_length(self, max_new_tokens: int = 0) -> int: ...
    @cached_property
    def retry_wrapper(self): ...
    def generate_batch(self, prompts: list[str], max_tokens: int | None = 4096, temperature: float = 0.7, **kwargs: Any) -> list[str]: ...
