from _typeshed import Incomplete
from ydata.synthesizers.text.__providers import AnthropicModel, OpenAIModel
from ydata.synthesizers.text.__providers.base import LLMProvider

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
    provider: Incomplete
    model_name: Incomplete
    def __init__(self, api_key: str | None = None, provider: LLMProvider | str = ..., model_name: OpenAIModel | AnthropicModel | str | None = None) -> None:
        """
        Initialize the base generator.

        Args:
            model_type (Union[ModelType, str]): Type of LLM to use
            model_name (Optional[str]): Specific model name to use

        Raises:
            ValueError: If model_type is invalid
        """
    def save(self, path: str):
        """
        Save the current configuration to a file.

        Args:
            path (str): Path where to save the configuration
        """
    @classmethod
    def load(cls, path: str) -> BaseGenerator:
        """
        Load a configuration from a file.

        Args:
            path (str): Path to the configuration file

        Returns:
            BaseGenerator: New instance with loaded configuration
        """
    def generate(self, *args, **kwargs) -> None:
        """
        Generate content based on input.
        Should be implemented by child classes.

        Raises:
            NotImplementedError: If not implemented by child class
        """
