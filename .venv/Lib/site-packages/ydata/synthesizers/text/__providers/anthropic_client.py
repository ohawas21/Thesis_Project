"""
    Anthropic API Client class definition
"""
from typing import Optional, Any, List

from enum import Enum
from functools import cached_property
import logging
from concurrent.futures import ThreadPoolExecutor

from tenacity import (
    retry,
    retry_if_exception_type,
    wait_exponential,
    stop_after_attempt,
    before_sleep_log,
    after_log,
)

import tiktoken
from anthropic import Anthropic

from ydata.synthesizers.text.__providers.base import BaseLLMClient, content_lengths

logger = logging.getLogger(__name__)

class AnthropicModel(str, Enum):
    CLAUDE_3_OPUS = "claude-3-opus-latest"
    CLAUDE_3_SONNET = "claude-3-7-sonnet-latest"
    CLAUDE_3_HAIKU = "claude-3-haiku-latest"

def estimate_batch_anthropic_cost(
    prompts=None,
    model=AnthropicModel.CLAUDE_3_SONNET,
    request_type='medium',
    response_type='medium',
    output_prompts=None,
    n_prompts: int | None = None,
):
    model = AnthropicModel(model)

    if request_type not in content_lengths:
        raise ValueError(f"Unsupported request type: {request_type}")
    if response_type not in content_lengths:
        raise ValueError(f"Unsupported response type: {response_type}")

    # Claude tokenizer not available; use GPT-3.5-turbo as approximation
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

    # Determine number of prompts and input tokens
    if prompts is not None:
        num_prompts = len(prompts)
        total_input_tokens = sum(len(enc.encode(prompt)) for prompt in prompts)
    else:
        if n_prompts is not None:
            total_input_tokens = n_prompts * content_lengths[request_type]
        else:
            raise ValueError("Either prompts or n_promtps must be specified.")

    # Determine output token count
    if output_prompts is not None:
        if prompts is not None and len(output_prompts) != len(prompts):
            raise ValueError("Length of output_prompts must match the number of prompts.")
        total_output_tokens = sum(len(enc.encode(output)) for output in output_prompts)
    else:
        total_output_tokens = num_prompts * content_lengths[response_type]

    total_tokens = total_input_tokens + total_output_tokens

    return {
        "model": model.value,
        "num_prompts": num_prompts,
        "input_tokens_total": total_input_tokens,
        "output_tokens_total": total_output_tokens,
        "total_tokens": total_tokens,
    }


class AnthropicClient(BaseLLMClient):
    def __init__(
        self,
        api_key: str,
        model: AnthropicModel = AnthropicModel.CLAUDE_3_SONNET,
        system_prompt: Optional[str] = None,
        chat_prompt_template: Optional[str] = None,
        max_workers: int = 4,
    ):
        super().__init__(
            system_prompt=system_prompt,
            chat_prompt_template=chat_prompt_template,
        )
        self.api_key = api_key
        self.model = AnthropicModel(model)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    @cached_property
    def client(self) -> Anthropic:
        return Anthropic(api_key=self.api_key)

    @cached_property
    def retry_wrapper(self):
        log = logging.getLogger("anthropic.retry")
        log.setLevel(logging.INFO)

        @retry(
            retry=retry_if_exception_type(Exception),
            wait=wait_exponential(multiplier=1, min=3, max=60),
            stop=stop_after_attempt(5),
            before_sleep=before_sleep_log(log, logging.INFO),
            after=after_log(log, logging.INFO),
            reraise=True,
        )
        def _wrapped(func, *args, **kwargs):
            return func(*args, **kwargs)

        return _wrapped

    def get_max_context_length(self, max_new_tokens: int = 0) -> int:
        max_token_limit = 200_000  # conservative max across Claude 3 models
        return max_token_limit - max_new_tokens

    def _call_api(
        self, prompt: str, max_tokens: int, temperature: float, **kwargs
    ) -> str:
        logger.debug(f"Calling Claude API with prompt tokens: {self.count_tokens(prompt)}")
        response = self.retry_wrapper(
            self.client.messages.create,
            model=self.model.value,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
            system=self.system_prompt,
            **kwargs,
        )
        return "".join(part.text for part in response.content).strip()

    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: Optional[int] = 4096,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> List[str]:
        logger.info(f"Generating batch of {len(prompts)} prompts")
        futures = [
            self.executor.submit(
                self._call_api,
                prompt,
                max_tokens or 4096,
                temperature,
                **kwargs
            ) for prompt in prompts
        ]
        return [f.result() for f in futures]
