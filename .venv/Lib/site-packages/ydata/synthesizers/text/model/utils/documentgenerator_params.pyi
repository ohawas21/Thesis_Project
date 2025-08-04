from dataclasses import dataclass
from enum import Enum
from typing import Any

class ToneCategory(Enum):
    FORMAL = 'formal'
    CASUAL = 'casual'
    PERSUASIVE = 'persuasive'
    EMPATHETIC = 'empathetic'
    INSPIRATIONAL = 'inspirational'
    ENTHUSIASTIC = 'enthusiastic'
    HUMOROUS = 'humorous'
    NEUTRAL = 'neutral'
    PROFESSIONAL = 'professional'

@dataclass
class DocumentGenerationParams:
    """
    Parameters for document generation.

    Attributes:
        document_type (str): Type of document to generate (required)
        audience (Optional[str]): Target audience for the document
        tone (Optional[str]): Desired tone of the document
        purpose (Optional[str]): Purpose of the document
        region (Optional[str]): Target region for the document
        language (Optional[str]): Language of the document
        length (Optional[str]): Desired length of the document
        topics (Optional[str]): Key points to include in the document
        style_guide (Optional[str]): Style guide to follow
    """
    document_type: str
    tone: str | None = ...
    audience: str | None = ...
    purpose: str | None = ...
    region: str | None = ...
    language: str | None = ...
    length: str | None = ...
    topics: str | None = ...
    style_guide: str | None = ...
    prompt_ending: str | None = ...
    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
    def to_dict(self) -> dict[str, Any]:
        """
        Convert parameters to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the parameters
        """
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocumentGenerationParams:
        """
        Create DocumentGenerationParams from a dictionary.

        Args:
            data (Dict[str, Any]): Dictionary containing parameter values

        Returns:
            DocumentGenerationParams: New instance with values from the dictionary
        """
    def __init__(self, document_type, tone=..., audience=..., purpose=..., region=..., language=..., length=..., topics=..., style_guide=..., prompt_ending=...) -> None: ...
