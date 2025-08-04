"""
    OpenAI API Client class definition
"""
import os
from enum import Enum

from typing import Optional, Any, List
from functools import cached_property
from concurrent.futures import ThreadPoolExecutor
import logging

import tiktoken
from openai import OpenAI
from openai import APIError, APIConnectionError, RateLimitError

from tenacity import (
    retry,
    retry_if_exception_type,
    wait_exponential,
    stop_after_attempt,
    before_sleep_log,
    after_log,
)

from ydata.synthesizers.text.__providers.base import BaseLLMClient, content_lengths
from ydata.synthesizers.text.__providers.base import BaseLLMClient

logger = logging.getLogger(__name__)

class OpenAIModel(str, Enum):
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_4 = "gpt-4"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4o ='gpt-4o'
    GPT_4o_mini='gpt-4o-mini'

def estimate_batch_openai_cost(
    prompts=None,
    model="gpt-4-turbo",
    request_type="medium",
    response_type="medium",
    output_prompts=None,
    n_prompts: int | None = None,
):
    model = OpenAIModel(model)

    if request_type not in content_lengths:
        raise ValueError(f"Unsupported request type: {request_type}")
    if response_type not in content_lengths:
        raise ValueError(f"Unsupported response type: {response_type}")

    # Load tokenizer for model
    enc = tiktoken.encoding_for_model(model.value)

    # Determine number of prompts and input token count
    if prompts is not None:
        n_prompts = len(prompts)
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
        total_output_tokens = n_prompts * content_lengths[response_type]

    total_tokens = total_input_tokens + total_output_tokens

    return {
        "model": model.value,
        "num_prompts": n_prompts,
        "input_tokens_total": total_input_tokens,
        "output_tokens_total": total_output_tokens,
        "total_tokens": total_tokens
    }


class OpenAIClient(BaseLLMClient):
    def __init__(
        self,
        api_key: str,
        model: OpenAIModel = OpenAIModel.GPT_3_5_TURBO,
        system_prompt: Optional[str] = None,
        chat_prompt_template: Optional[str] = None,
        max_workers: int = 4,
    ):
        super().__init__(system_prompt=system_prompt, chat_prompt_template=chat_prompt_template)
        self.api_key = api_key
        self.model = model
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    @cached_property
    def client(self) -> OpenAI:
        return OpenAI(api_key=self.api_key)

    def get_max_context_length(self, max_new_tokens: int = 0) -> int:
        context_limits = {
            OpenAIModel.GPT_3_5_TURBO: 4096,
            OpenAIModel.GPT_4: 8192,
            OpenAIModel.GPT_4_TURBO: 128_000,
            OpenAIModel.GPT_4o: 128_000,
            OpenAIModel.GPT_4o_mini: 128_000
        }
        return context_limits[self.model] - max_new_tokens

    @cached_property
    def retry_wrapper(self):
        log = logging.getLogger("openai.retry")
        log.setLevel(logging.INFO)

        @retry(
            retry=retry_if_exception_type((
                RateLimitError,
                APIError,
                APIConnectionError,
            )),
            wait=wait_exponential(multiplier=1, min=2, max=60),
            stop=stop_after_attempt(5),
            before_sleep=before_sleep_log(log, logging.INFO),
            after=after_log(log, logging.INFO),
            reraise=True,
        )
        def _wrapped(func, *args, **kwargs):
            return func(*args, **kwargs)

        return _wrapped

    def _call_api(
        self,
        prompt: str,
        max_tokens: Optional[int],
        temperature: float,
        **kwargs: Any
    ) -> str:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        logger.debug(f"Calling OpenAI API with {len(messages)} messages")

        response = self.retry_wrapper(
            self.client.chat.completions.create,
            model=self.model.value,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        return response.choices[0].message.content.strip()

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
                max_tokens,
                temperature,
                **kwargs
            ) for prompt in prompts
        ]
        return [f.result() for f in futures]


