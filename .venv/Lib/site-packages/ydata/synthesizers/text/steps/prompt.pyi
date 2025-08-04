import pyarrow as pa
from _typeshed import Incomplete
from typing import Callable
from ydata.synthesizers.text.__providers import AnthropicModel, OpenAIModel

class Prompt:
    post_process: Incomplete
    system_prompt: Incomplete
    client: Incomplete
    def __init__(self, client: OpenAIModel | AnthropicModel, system_prompt: str | None = None, post_process: Callable[[str], str] | None = None) -> None: ...
    def generate(self, dataset: pa.Table, max_tokens: int | None = None, temperature: float = 0.7, prompt_column: str = 'prompt', output_column: str = 'generations', **kwargs) -> pa.Table | list[str]: ...
