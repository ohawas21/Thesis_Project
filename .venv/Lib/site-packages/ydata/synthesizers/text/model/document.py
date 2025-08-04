"""
    Synthetic document generator that creates documents in different formats (DOCX, PDF, HTML)
    based on JSON input specifications.
"""
from enum import Enum
from typing import Union, Optional, Dict, Any, List
import os
import tempfile
from random import choice

import pyarrow as pa

from ydata.synthesizers.text.model.base import BaseGenerator
from ydata.synthesizers.text.__providers import (LLMProvider, OpenAIModel, AnthropicModel,
                                                estimate_batch_anthropic_cost, estimate_batch_openai_cost)

from ydata.synthesizers.text.model.prompts.document_type import DocumentType

from ydata.synthesizers.text.model.prompts.document import (GENERAL_PURPOSE_DOCUMENTS_TOPICS,
                                                            GENERAL_PURPOSE_DOCUMENTS_HTML,
                                                            GENERAL_PURPOSE_HTML_SYSTEM_PROMPT,
                                                            DOCUMENT_GENERATION_PROMPT_ENDINGS,
                                                            DOCUMENTS_TONE_POOL)

from ydata.synthesizers.text.steps.prompt import Prompt
from ydata.synthesizers.text.model.utils.documentgenerator_params import ToneCategory, DocumentGenerationParams
from ydata.synthesizers.text.model.utils.render_html import (clean_html, render_html_to_pdf, save_html, render_html_to_docx)

from ydata.dataset.textdataset import TextDataset

from ydata.utils.logger import DATATYPE_MAPPING, SDKLogger

from ydata._licensing import llm_tokens_check, llm_tokens_charge

metrics_logger = SDKLogger(name="Metrics logger")

class DocumentFormat(Enum):
    """
    Enum representing supported output formats for synthetic document generation.

    Attributes:
        DOCX: Microsoft Word document format (docx)
        PDF: Portable Document Format (pdf)
        HTML: HyperText Markup Language format (html)
    """
    DOCX = "docx"
    PDF = "pdf"
    HTML = "html"

def map_topic_prompts(row):
    trf_row = row.copy()
    custom_prompt = DocumentType.get_content_prompt(row['document_type'])
    prompt_template = custom_prompt or GENERAL_PURPOSE_DOCUMENTS_TOPICS

    trf_row['prompt_content'] = prompt_template.format(
        document_type=row['document_type'],
        audience=row['audience'],
        tone=row['tone'],
        purpose=row['purpose'],
        region=row['region'],
        language=row['language'],
        length=row['length'],
        topics=row['topics'],
        style_guide=row['style_guide'],
        prompt_ending=row['prompt_ending'],
    ).strip()
    return trf_row

def map_html_prompts(row):
    trf_row = row.copy()
    custom_prompt = DocumentType.get_html_prompt(row['document_type'])
    prompt_template = custom_prompt or GENERAL_PURPOSE_DOCUMENTS_HTML

    trf_row['prompt_html'] = prompt_template.format(
                    document_type=row['document_type'],
                    audience=row['audience'],
                    tone=row['tone'],
                    document_content=row['documents_content']
                ).strip()
    return trf_row

def map_clean_html(row):
    trf_row = row.copy()
    trf_row['html'] = clean_html(row['generations'])
    return trf_row

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

    def __init__(
        self,
        provider: Union[LLMProvider, str] = LLMProvider.OPENAI,
        model_name: Optional[Union[OpenAIModel, AnthropicModel, str]] = OpenAIModel.GPT_4,
        document_format: Optional[Union[DocumentFormat, str]] = DocumentFormat.PDF,
        api_key: Optional[str] = None,
    ):
        super().__init__(api_key=api_key, provider=provider, model_name=model_name)
        self.document_format = DocumentFormat(document_format)

    def _generate_documents_content(self, documents_specs: TextDataset):
        """
        Method responsible to generate the documents given the user input
        Returns:
            A TextDataset
        """
        documents_content_prompt = Prompt(
            client=self._client,
            system_prompt="You are a document content generator. Generate high-quality, coherent content based on the given specifications."
        )

        documents_topic_prompts = documents_specs.map(
            map_topic_prompts,
            name="Mapped topic prompts"
        ).to_pyarrow()

        # Here call the models with the prompt defined
        documents_content = documents_content_prompt.generate(documents_topic_prompts,
                                                              output_column='documents_content',
                                                              prompt_column='prompt_content')

        return documents_content

    def _generate_documents_html(self, documents_content: TextDataset):
        """
        Method responsible to generate the documents HTML structure and display given the document type and expected format.
        Returns:
            A TextDataset with the HTML format
        """
        # Have this separated into a private function
        documents_html_prompt = Prompt(
            client=self._client,
            system_prompt=GENERAL_PURPOSE_HTML_SYSTEM_PROMPT
        )

        documents_html_prompts = documents_content.map(
            map_html_prompts,
            name="Mapped html prompts"
        ).to_pyarrow()

        documents_html = documents_html_prompt.generate(documents_html_prompts,
                                                        prompt_column='prompt_html')

        documents_html = documents_html.map(
            map_clean_html,
            name="Clean generated HTML code"
        )

        return documents_html

    def _render_documents(self,
                          documents_html: list,
                          output_path: str):
        """
        Method responsible to render the documents given the document type and expected format.
        Args:
            documents_html: A TextDataset with the HTML content to be rendered into different formats
            document_format: The expected output format (DocumentFormat)
        """

        if self.document_format == DocumentFormat.DOCX:
            render_html_to_docx(documents_html, output_path)
        elif self.document_format == DocumentFormat.PDF:
            render_html_to_pdf(documents_html, output_path)
        elif self.document_format == DocumentFormat.HTML:
            save_html(documents_html, output_path) #save the files to HTML

    def __calc_tokens(self, input_prompts: list | None=None,
                      output_prompts: list | None=None,
                      n_prompts: int | None=None) -> dict:
        """
        Method that calculated the tokens sent and received from the models
        Args:
            input_prompts: A list with the prompts sent to the llm model
            output_prompts: A list with the prompts received from the llm model

        Returns: a dictionary with the information regarding the number of inputs and associated costs
        """
        if (input_prompts is None) and (output_prompts is None):
            # --- Pre-generation token estimation ---
            if self.provider == LLMProvider.OPENAI:
                cost = estimate_batch_openai_cost(
                    model=self.model_name,
                    request_type="long",
                    response_type="ultra_long",
                    n_prompts=n_prompts,
                )
            elif self.provider == LLMProvider.ANTHROPIC:
                cost = estimate_batch_anthropic_cost(
                    prompts=input_prompts,
                    model=self.model_name,
                    request_type="long",
                    response_type="ultra_long",
                    n_prompts=n_prompts,
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
        document_type: str | None = None,
        n_docs: int = 1,
        audience: str | None = None,
        tone: str | ToneCategory | None = None,
        purpose: str | None = None,
        region: str | None = None,
        language: str | None = None,
        length: str | None = None,
        topics: str | None = None,
        style_guide: str | None = None,
        output_dir: Optional[str] = None,
        **kwargs
    ) -> List[str]:
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

        # Initialize output directory
        if output_dir is None:
            output_dir = tempfile.mkdtemp()
        os.makedirs(output_dir, exist_ok=True)

        metrics_logger.info(dataset=None,
                            datatype=DATATYPE_MAPPING[self.__class__.__name__],
                            method='documentgen',
                            ndocs=n_docs)

        # Load and validate specifications
        specs = self._prepare_specifications(
            document_type=document_type,
            n_docs=n_docs,
            audience=audience,
            tone=tone,
            purpose=purpose,
            region=region,
            language=language,
            length=length,
            topics=topics,
            style_guide=style_guide
        )

        #Pre-check if the user has enough credits
        pre_costs = self.__calc_tokens(n_prompts=n_docs)

        # Send to backoffice to check if the user has enough credits
        llm_tokens_check(service=self.provider.value, costs_info=pre_costs)

        specs = TextDataset(data=pa.Table.from_pylist(specs),
                            name="Document specs")

        documents_content = self._generate_documents_content(specs)

        documents_html = self._generate_documents_html(documents_content)

        #Check the actual amount of credits to be charged
        docs_table = documents_html.to_pyarrow()
        costs = self.__calc_tokens(input_prompts=docs_table['prompt_content'].to_pylist()+docs_table['prompt_html'].to_pylist(),
                                   output_prompts=docs_table['documents_content'].to_pylist()+docs_table['generations'].to_pylist())

        # Send to backoffice to charge the user
        llm_tokens_charge(service=self.provider.value, costs_info=costs)

        self._render_documents(docs_table.to_pylist(),
                               output_path=output_dir)

    def _prepare_specifications(
        self,
        document_type: str,
        audience: str | None = None,
        tone: str | None = None,
        purpose: str | None = None,
        region: str | None = None,
        language: str | None = None,
        length: str | None = None,
        topics: str | None = None,
        style_guide: str | None = None,
        n_docs: int = 1
    ) -> List[Dict[str, Any]]:
        """Prepare document specifications from either individual parameters or documents_request."""
        length_pool = ["short", "medium", "long", "brief", "detailed"]

        if tone is not None:
            tone = ToneCategory(tone).value
            tone_pool = DOCUMENTS_TONE_POOL[tone]
        else:
            tone_pool = DOCUMENTS_TONE_POOL[ToneCategory.FORMAL.value]

        # Use synonym clusters to group tones/purposes that are similar
        base_tone = tone or choice(tone_pool)
        base_length = length or choice(length_pool)
        prompt_ending = choice(DOCUMENT_GENERATION_PROMPT_ENDINGS)

        specs =[]
        try:
            for _ in range(n_docs):
                params = DocumentGenerationParams(
                    document_type=document_type,
                    tone=base_tone,
                    audience=audience,
                    purpose=purpose,
                    region=region,
                    language=language,
                    length=base_length,
                    topics=topics,
                    style_guide=style_guide,
                    prompt_ending=prompt_ending
                ).to_dict()

                specs.append(params)
        except ValueError as e:
            raise ValueError(f"Invalid document parameters: {str(e)}")

        return specs
