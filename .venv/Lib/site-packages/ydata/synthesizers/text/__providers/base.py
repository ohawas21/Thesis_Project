"""
    Base class for providers API calls
"""
from enum import Enum
from typing import Any, Optional, Union, List
from abc import ABC, abstractmethod

from itertools import chain

import tiktoken
import pyarrow as pa

class LLMProvider(Enum):
    """Enum for supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"

# Output token estimate by response type
content_lengths = {
    "very_short": 100,      # Ideal for headlines, outlines, or single-sentence responses (e.g., step 1 prompts)
    "short": 300,           # Great for short intros, summaries, or small paragraphs
    "medium": 700,          # Good for basic articles, product descriptions, or concise emails
    "long": 1200,           # Suitable for multi-paragraph content like blog posts, essays, or brief reports
    "very_long": 1800,      # Use for detailed content, such as in-depth guides, proposals, or long-form answers
    "ultra_long": 2500,     # Extended documents — e.g., full blog articles, onboarding documents, or light whitepapers
    "mega": 5000,           # Multi-section documents like research reports, eBooks, or full technical documentation
}

class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        chat_prompt_template: Optional[str] = None,
    ):
        self.system_prompt = system_prompt
        self.chat_prompt_template = chat_prompt_template

    @abstractmethod
    def get_max_context_length(self, max_new_tokens: int) -> int:
        pass

    def generate(
        self,
        table: pa.Table,
        prompt_column: str = "prompt",
        in_context_examples: Optional[list[str]] = None,
        end_instruction: Optional[str] = None,
        sep: str = "\n",
        min_in_context_examples: Optional[int] = None,
        max_in_context_examples: Optional[int] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        return_table: bool = False,
        **kwargs: Any,
    ) -> Union[List[str], pa.Table]:
        """
        Main user-facing generation method. Handles steps formatting and generation.
        Takes a pyarrow.Table as input and applies batch generation based on a specified column.
        """

        if prompt_column not in table.column_names:
            raise ValueError(f"'{prompt_column}' column not found in input table.")

        prompts = []
        for row in table.to_pydict()[prompt_column]:
            # Step 1: Format steps
            prompt = self._format_prompt(
                beg_instruction=row,
                in_context_examples=in_context_examples,
                end_instruction=end_instruction,
                sep=sep,
                min_in_context_examples=min_in_context_examples,
                max_in_context_examples=max_in_context_examples,
            )

            # Step 2: Apply chat template if defined
            final_prompt = self._apply_chat_template(prompt)
            prompts.append(final_prompt)

        # Step 3: Generate responses in batch
        responses = self.generate_batch(
            prompts=prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

        return responses

    def _format_prompt(
        self,
        beg_instruction: Optional[str],
        in_context_examples: Optional[list[str]],
        end_instruction: Optional[str],
        sep: str,
        min_in_context_examples: Optional[int],
        max_in_context_examples: Optional[int],
    ) -> str:
        in_context_examples = in_context_examples or []

        if len(in_context_examples) > 0:
            if min_in_context_examples is None:
                min_in_context_examples = 1
            if max_in_context_examples is None:
                max_in_context_examples = len(in_context_examples)

            assert min_in_context_examples >= 0
            assert max_in_context_examples >= min_in_context_examples
            assert len(in_context_examples) >= min_in_context_examples

        selected_examples = in_context_examples[:max_in_context_examples] \
            if max_in_context_examples is not None else in_context_examples

        if (
            min_in_context_examples is not None and
            len(selected_examples) < min_in_context_examples
        ):
            raise ValueError(
                f"Cannot fit the minimum {min_in_context_examples} in-context examples."
            )

        parts = list(chain(
            [beg_instruction] if beg_instruction else [],
            selected_examples,
            [end_instruction] if end_instruction else []
        ))

        return sep.join(parts)

    def _apply_chat_template(self, prompt: str) -> str:
        if not self.chat_prompt_template:
            return prompt
        return (
            self.chat_prompt_template
            .replace("{{system_prompt}}", self.system_prompt or "")
            .replace("{{steps}}", prompt)
        )

    def count_tokens(self, text: str, model_name: Optional[str] = None) -> int:

        if model_name is None:
            enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        else:
            enc = tiktoken.encoding_for_model(model_name)
        return len(enc.encode(text))

    def unload_model(self):
        pass
