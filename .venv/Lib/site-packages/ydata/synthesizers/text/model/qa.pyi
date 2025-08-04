import pyarrow as pa
from _typeshed import Incomplete
from langchain.schema import Document as Document
from ydata.synthesizers.text.__providers import AnthropicModel, OpenAIModel
from ydata.synthesizers.text.__providers.base import LLMProvider
from ydata.synthesizers.text.model.base import BaseGenerator

metrics_logger: Incomplete

class DocumentQAGeneration(BaseGenerator):
    """
    A class for generating Question-Answer pairs from documents using Large Language Models.
    Inherits from BaseGenerator for common LLM functionality.

    Features:
        - Support for multiple document formats (DOCX, TXT)
        - Batch processing of multiple documents
        - Configurable LLM selection
        - Persistent configuration saving/loading
        - PyArrow integration for efficient data handling
        - LangChain integration for document processing and chunking

    Args:
        model_type (ModelType): The type of LLM to use
        model_name (str, optional): Specific model name to use
        chunk_size (int): Size of text chunks for processing
        chunk_overlap (int): Overlap between chunks
        document_type (DocumentType, optional): Type of document being processed
    """
    chunk_size: Incomplete
    chunk_overlap: Incomplete
    def __init__(self, provider: LLMProvider | str = ..., model_name: OpenAIModel | AnthropicModel | str | None = None, chunk_size: int = 1000, chunk_overlap: int = 200, api_key: str | None = None) -> None:
        """
        Initialize the DocumentQAGeneration class.

        Args:
            model_type (Union[ModelType, str]): Type of LLM to use
            model_name (str, optional): Specific model name to use
            chunk_size (int): Size of text chunks for processing
            chunk_overlap (int): Overlap between chunks
            document_type (DocumentType, optional): Type of document being processed
        """
    def generate(self, input_source: str | pa.Table, docs_extension: str = 'docx', num_qa_pairs: int = 10) -> pa.Table:
        """
        Generate Q&A scenarios from documents.

        Args:
            input_source: Either a path to a document/folder or a pyarrow Table
            docs_extension: Extension of documents to process
            num_qa_pairs: Number of Q&A pairs to generate
            output_dir: Directory to store intermediate results

        Returns:
            pa.Table: PyArrow table containing the generated Q&A pairs

        Raises:
            ValueError: If input_source is invalid
        """
