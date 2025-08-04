"""
    Base class for all generators in the project.
    Provides common functionality and interfaces for document and Q&A generation.
"""
import json
from pathlib import Path
from typing import Union, Optional, Dict, Any

from ydata.synthesizers.text.__providers.base import LLMProvider
from ydata.synthesizers.text.__providers import OpenAIModel, AnthropicModel, OpenAIClient, AnthropicClient

from ydata._aiservicetokens import openai as apikey_openai, anthropic as apikey_anthropic #needed to use alias to avoid
                                                                                         # any confusion with openai methods which have the same name

class BaseGenerator:
    """
    Base class for all generators in the project.
    Provides common functionality for model initialization and configuration management.

    Attributes:
        model_type (ModelType): The type of LLM to use
        model_name (Optional[str]): Specific model name to use
        llm: The initialized language model instance

    Example:
        >>> class CustomGenerator(BaseGenerator):
        ...     def __init__(self, provider: LLMProvider = LLMProvider.OPENAI):
        ...         super().__init__(provider)
        ...     def generate(self, input_data):
        ...         # Implementation specific to this generator
    """
    def __init__(
        self,
        api_key: str | None = None,
        provider: Union[LLMProvider, str] = LLMProvider.OPENAI,
        model_name: Optional[Union[OpenAIModel, AnthropicModel, str]] = None,
    ):
        """
        Initialize the base generator.

        Args:
            model_type (Union[ModelType, str]): Type of LLM to use
            model_name (Optional[str]): Specific model name to use

        Raises:
            ValueError: If model_type is invalid
        """
        # Convert string to enum if necessary
        self.provider = LLMProvider(provider)

        if self.provider == LLMProvider.OPENAI:
            if model_name is None:
                self.model_name = OpenAIModel.GPT_4
            else:
                self.model_name = OpenAIModel(model_name)

            if api_key is None:
                api_key = apikey_openai()
            self._client = OpenAIClient(api_key=api_key,
                                        model=self.model_name)
        else:
            if model_name is None:
                self.model_name = AnthropicModel.CLAUDE_3_SONNET
            else:
                self.model_name = AnthropicModel(model_name)

            if api_key is None:
                api_key = apikey_anthropic()
            self._client = AnthropicClient(api_key=api_key,
                                           model=self.model_name)

    def save(self, path: str):
        """
        Save the current configuration to a file.

        Args:
            path (str): Path where to save the configuration
        """
        config_path = Path(path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Create configuration dictionary from current attributes
        save_config = {
            "model_type": self.model_type.value,
            "model_name": self.model_name.value,
        }

        # Add any additional configuration from child classes
        save_config.update(self._get_additional_config())

        with open(config_path, 'w') as f:
            json.dump(save_config, f, indent=4)

    @classmethod
    def load(cls, path: str) -> 'BaseGenerator':
        """
        Load a configuration from a file.

        Args:
            path (str): Path to the configuration file

        Returns:
            BaseGenerator: New instance with loaded configuration
        """
        with open(path, 'r') as f:
            config = json.load(f)

        # Create instance with basic config
        instance = cls(
            model_type=config["model_type"],
            model_name=config["model_name"],
        )

        # Load any additional configuration
        instance._load_additional_config(config)

        return instance

    def _get_additional_config(self) -> Dict[str, Any]:
        """
        Get additional configuration from child classes.
        Should be overridden by child classes that need to save additional state.

        Returns:
            Dict[str, Any]: Additional configuration to save
        """
        return {}

    def _load_additional_config(self, config: Dict[str, Any]):
        """
        Load additional configuration in child classes.
        Should be overridden by child classes that need to load additional state.

        Args:
            config (Dict[str, Any]): Configuration to load
        """
        pass

    def generate(self, *args, **kwargs):
        """
        Generate content based on input.
        Should be implemented by child classes.

        Raises:
            NotImplementedError: If not implemented by child class
        """
        raise NotImplementedError("Subclasses must implement generate()")
