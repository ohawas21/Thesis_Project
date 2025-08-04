from _typeshed import Incomplete
from enum import Enum
from ydata.synthesizers.text.__providers import AnthropicModel, LLMProvider, OpenAIModel
from ydata.synthesizers.text.model.base import BaseGenerator
from ydata.synthesizers.text.model.utils.documentgenerator_params import ToneCategory

metrics_logger: Incomplete

class DocumentFormat(Enum):
    """
    Enum representing supported output formats for synthetic document generation.

    Attributes:
        DOCX: Microsoft Word document format (docx)
        PDF: Portable Document Format (pdf)
        HTML: HyperText Markup Language format (html)
    """
    DOCX = 'docx'
    PDF = 'pdf'
    HTML = 'html'

def map_topic_prompts(row): ...
def map_html_prompts(row): ...
def map_clean_html(row): ...

class DocumentGenerator(BaseGenerator):
    """
    A class for generating synthetic documents in various formats (DOCX, PDF, HTML)
    based on input specifications.

    Features:
        - Support for multiple document formats (DOCX, PDF, HTML)
        - Configurable LLM selection
        - Template-based document generation
        - Customizable document structure and styling
        - Batch processing of multiple document specifications

    Args:
        api_key (str): API key for the LLM provider
        provider (Union[LLMProvider, str]): The LLM provider to use
        model_name (Optional[Union[OpenAIModel, AnthropicModel, str]]): Specific model to use
        default_format (DocumentFormat): Default output format if not specified in request
    """
    document_format: Incomplete
    def __init__(self, provider: LLMProvider | str = ..., model_name: OpenAIModel | AnthropicModel | str | None = ..., document_format: DocumentFormat | str | None = ..., api_key: str | None = None) -> None: ...
    def generate(self, document_type: str | None = None, n_docs: int = 1, audience: str | None = None, tone: str | ToneCategory | None = None, purpose: str | None = None, region: str | None = None, language: str | None = None, length: str | None = None, topics: str | None = None, style_guide: str | None = None, output_dir: str | None = None, **kwargs) -> list[str]:
        """
        Generate documents based on input specifications.

        Args:
            document_type (str, optional): Type of document to generate
            audience (str, optional): Target audience for the document
            tone (str, optional): Desired tone of the document. Can be selected from the following limited list of values formal, casual, persuasive, empathetic, inspirational, enthusiastic, humorous, neutral.
            purpose (str, optional): Purpose of the document
            region (str, optional): Target region/locale
            language (str, optional): Language of the document
            length (str, optional): Desired length of the document
            topics (str, optional): Key points to cover
            style_guide (str, optional): Style guide to follow
            output_dir (str, optional): Directory to store generated documents
            **kwargs: Additional arguments to pass to the generation process

        Raises:
            ValueError: If input validation fails or document format is unsupported
        """
