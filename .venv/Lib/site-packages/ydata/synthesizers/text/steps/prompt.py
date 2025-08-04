"""
    File that define the abstraction layer that orchestrates between different LLM clients
"""
from copy import deepcopy

import pyarrow as pa
from typing import Optional, Union, Callable, List

from ydata.dataset import TextDataset
from ydata.synthesizers.text.__providers import (AnthropicModel,
                                                 OpenAIModel)

class Prompt:
    def __init__(
        self,
        client: Union[OpenAIModel, AnthropicModel],
        system_prompt: Optional[str] = None,
        post_process: Optional[Callable[[str], str]] = None,
    ):
        self.post_process = post_process
        self.system_prompt = system_prompt

        self.client = client

    def generate(
        self,
        dataset: pa.Table,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        prompt_column: str = "prompt",
        output_column: str = "generations",
        **kwargs
    ) -> Union[pa.Table, List[str]]:

        if prompt_column not in dataset.column_names:
            raise ValueError(f"Column '{prompt_column}' not found in input table.")

        prompts = deepcopy(dataset).select([prompt_column])

        # Call the LLM client
        generations = self.client.generate(
            table=prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            prompt_column=prompt_column,
            **kwargs
        )

        # Optional post-processing
        if self.post_process:
            generations = [self.post_process(g) for g in generations]

        return TextDataset(name="Generated outputs",
                           data=dataset.append_column(output_column, pa.array(generations)))
