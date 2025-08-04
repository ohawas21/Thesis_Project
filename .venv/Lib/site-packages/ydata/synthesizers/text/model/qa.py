"""
    Synthetic data generator for Q&A pairs
"""
from typing import Union, Optional, List, Dict, Any

import ast
import pandas as pd
import pyarrow as pa

try:
    from langchain_community.document_loaders import UnstructuredWordDocumentLoader
    from langchain_community.document_loaders import (
        TextLoader,
        DirectoryLoader
    )
    from langchain.schema import Document
except ImportError:
    raise ImportError(
        f"The 'ydata.synthesizers.text.model.qa.DocumentQAGeneration' module requires the following packages: langchain" +
        ". Please install with: pip install \"ydata-sdk[text, docx]\""
    )

from ydata.dataset import TextDataset

from ydata.synthesizers.text.model.base import BaseGenerator
from ydata.synthesizers.text.__providers.base import LLMProvider
from ydata.synthesizers.text.__providers import (OpenAIModel, AnthropicModel,
                                                 estimate_batch_anthropic_cost, estimate_batch_openai_cost)
from ydata.synthesizers.text.utils import _validate_input_path
from ydata.synthesizers.text.model.prompts.qa import OUTPUT_REQUEST, QA_GENERATION_GENERAL
from ydata.synthesizers.text.steps.prompt import Prompt

from ydata.utils.logger import DATATYPE_MAPPING, SDKLogger

from ydata._licensing import llm_tokens_check, llm_tokens_charge

metrics_logger = SDKLogger(name="Metrics logger")

def _clean_output(answers: list,) -> pd.DataFrame:
    df = answers.to_pandas()
    all_qas = []
    for i, row in df.iterrows():
        text = row['generations']
        print(text)
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, dict):
                parsed = [parsed]
            for item in parsed:
                item['source'] = row['source']
                all_qas.append(item)
        except Exception:
            continue
    return pd.DataFrame(all_qas).drop_duplicates(subset=['source', 'question', 'answer'])

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

    def __init__(
        self,
        provider: Union[LLMProvider, str] = LLMProvider.OPENAI,
        model_name: Optional[Union[OpenAIModel, AnthropicModel, str]] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the DocumentQAGeneration class.

        Args:
            model_type (Union[ModelType, str]): Type of LLM to use
            model_name (str, optional): Specific model name to use
            chunk_size (int): Size of text chunks for processing
            chunk_overlap (int): Overlap between chunks
            document_type (DocumentType, optional): Type of document being processed
        """
        super().__init__(provider=provider, model_name=model_name, api_key=api_key)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def __load_data(self, input_source: Union[str, pa.Table], docs_extension: str):
        """Load and process input data."""

        if docs_extension not in ['docx', 'txt']:
            raise ValueError("Unsupported file type. Your input file type must be docx or txt.")

        if isinstance(input_source, str):
            is_file, psuffix = _validate_input_path(input_source)

            if is_file:
                suffix = psuffix
                documents = self._process_document(doc_path=input_source, suffix=suffix)
            else:
                suffix = docs_extension
                documents = self._process_folder(str(input_source), suffix=suffix)
            data = [{"source": doc.metadata['source'], "text": doc.page_content} for doc in documents]
        else:
            data = [{"source": m[1], 'text': text} for m, text in
                    zip(input_source.column('metadata'), input_source.column('text'))]

        return TextDataset(name='Load data files', data=data) #review how is this working

    def __calc_tokens(self, input_prompts: list | None=None, output_prompts: list | None=None)->dict:
        """
        Method that calculated the tokens sent and received from the models
        Args:
            input_prompts: A list with the prompts sent to the llm model
            output_prompts: A list with the prompts received from the llm model

        Returns: a dictionary with the information regarding the number of inputs and associated costs
        """
        if output_prompts is None:
            # --- Pre-generation token estimation ---
            if self.provider == LLMProvider.OPENAI:
                cost = estimate_batch_openai_cost(
                    prompts=input_prompts,
                    model=self.model_name,
                    response_type="medium",
                )
            elif self.provider == LLMProvider.ANTHROPIC:
                cost = estimate_batch_anthropic_cost(
                    prompts=input_prompts,
                    model=self.model_name,
                    response_type="medium",
                )
        else:
            if self.provider == LLMProvider.OPENAI:
                cost = estimate_batch_openai_cost(
                    prompts=input_prompts,
                    model=self.model_name,
                    output_prompts=output_prompts
                )
            elif self.provider == LLMProvider.ANTHROPIC:
                cost = estimate_batch_anthropic_cost(
                    prompts=input_prompts,
                    model=self.model_name,
                    output_prompts=output_prompts
                )
        return cost

    def generate(
        self,
        input_source: Union[str, pa.Table],
        docs_extension: str = 'docx',
        num_qa_pairs: int = 10,) -> pa.Table:
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
        #init the prompt
        prompt = Prompt(
            client = self._client,
            system_prompt="You're a questions and answers specialist analyzing documents"
        )

        documents = self.__load_data(input_source=input_source,
                                     docs_extension=docs_extension)

        metrics_logger.info(dataset=documents,
                            datatype=DATATYPE_MAPPING[self.__class__.__name__],
                            method='documentqa',
                            ndocs=documents.nrows)

        # Pre-calculate the cost to allow the API calls
        pre_cost = self.__calc_tokens(input_prompts=documents.to_pyarrow()['text'].to_pylist())

        # Send to backoffice to check if the user has enough credits
        llm_tokens_check(service=self.provider.value, costs_info=pre_cost)

        def map_prompts(row):
            trf_row = row.copy()
            trf_row['prompt'] = QA_GENERATION_GENERAL.format(info_json=row['text'],output_request=OUTPUT_REQUEST)
            return trf_row

        documents_prompt = documents.map(
            map_prompts,
            name="Mapped prompts"
        ).to_pyarrow()

        qa_dataset = prompt.generate(documents_prompt)

        #Calculate the actual cost
        costs = self.__calc_tokens(input_prompts=qa_dataset.to_pyarrow()['prompt'].to_pylist(),
                           output_prompts=qa_dataset.to_pyarrow()['generations'].to_pylist())

        # Send to backoffice to charge the user
        llm_tokens_charge(service=self.provider.value, costs_info=costs)

        qa_dataset = _clean_output(qa_dataset)

        return qa_dataset

    def _process_folder(self, folder_path: str, suffix: str) -> pa.Table:
        """
        Process all documents in a folder using LangChain's DirectoryLoader.

        Args:
            folder_path (str): Path to the folder containing documents
            suffix (str): File extension to process

        Returns:
            pa.Table: PyArrow table containing processed documents
        """
        loader = DirectoryLoader(
            folder_path,
            glob=f"**/*{suffix}",
            show_progress=True
        )
        documents = loader.load()
        return documents

    def _process_document(self, doc_path: str, suffix: str) -> pa.Table:
        """
        Process a single document using appropriate LangChain loader.

        Args:
            doc_path (str): Path to the document
            suffix (str): File extension

        Returns:
            pa.Table: PyArrow table containing processed document
        """
        if suffix == '.docx':
            loader = UnstructuredWordDocumentLoader(doc_path)
        elif suffix == '.txt':
            loader = TextLoader(doc_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        documents = loader.load()
        return documents

    def _process_documents(self, documents: List[Document]) -> pa.Table:
        """
        Process a list of documents into chunks.

        Args:
            documents (List[Document]): List of documents

        Returns:
            pa.Table: PyArrow table containing processed documents
        """
        chunks = self.text_splitter.split_documents(documents)
        texts = [chunk.page_content for chunk in chunks]
        metadata = [chunk.metadata for chunk in chunks]

        return pa.Table.from_pydict({
            "text": texts,
            "metadata": metadata
        })

    def _get_additional_config(self) -> Dict[str, Any]:
        """Get additional configuration specific to DocumentQAGeneration."""
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "document_type": self.document_type.value if self.document_type else None
        }

    def _load_additional_config(self, config: Dict[str, Any]):
        """Load additional configuration specific to DocumentQAGeneration."""
        self.chunk_size = config.get("chunk_size", 1000)
        self.chunk_overlap = config.get("chunk_overlap", 200)
        if config.get("document_type"):
            self.document_type = DocumentType(config["document_type"])
