"""
    Data class for the Document generator expected inputs
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional

from enum import Enum

class ToneCategory(Enum):
    FORMAL = "formal"
    CASUAL = "casual"
    PERSUASIVE = "persuasive"
    EMPATHETIC = "empathetic"
    INSPIRATIONAL = "inspirational"
    ENTHUSIASTIC = "enthusiastic"
    HUMOROUS = "humorous"
    NEUTRAL = "neutral"
    PROFESSIONAL = "professional"

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
    tone: str | None = None
    audience: Optional[str] = None
    purpose: Optional[str] = None
    region: Optional[str] = None
    language: Optional[str] = None
    length: Optional[str] = None
    topics: Optional[str] = None
    style_guide: Optional[str] = None
    prompt_ending: Optional[str] = None

    def __post_init__(self):
        """Validate parameters after initialization."""
        self._validate_document_type()
        self._validate_optional_fields()

    def _validate_document_type(self) -> None:
        """Validate the required document_type field."""
        if not isinstance(self.document_type, str):
            raise ValueError("Document type must be a string.")
        if not self.document_type.strip():
            raise ValueError("Document types is a mandatory input. It can't be None or an empty string.")

    def _validate_optional_fields(self) -> None:
        """Validate all optional string fields."""
        optional_fields = {
            'audience': self.audience,
            'tone': self.tone,
            'purpose': self.purpose,
            'region': self.region,
            'language': self.language,
            'length': self.length,
            'topics': self.topics,
            'style_guide': self.style_guide,
            'prompt_ending': self.prompt_ending,
        }

        for field_name, value in optional_fields.items():
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{field_name} must be a string or None")
            if value is not None and not value.strip():
                raise ValueError(f"{field_name} cannot be an empty string")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert parameters to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the parameters
        """
        return {
            'document_type': self.document_type,
            'audience': self.audience,
            'tone': self.tone,
            'purpose': self.purpose,
            'region': self.region,
            'language': self.language,
            'length': self.length,
            'topics': self.topics,
            'style_guide': self.style_guide,
            'prompt_ending': self.prompt_ending,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentGenerationParams':
        """
        Create DocumentGenerationParams from a dictionary.

        Args:
            data (Dict[str, Any]): Dictionary containing parameter values

        Returns:
            DocumentGenerationParams: New instance with values from the dictionary
        """
        if 'document_type' not in data:
            raise ValueError("document_type is required")
        return cls(**data)

    def __str__(self) -> str:
        """Return string representation of the parameters."""
        return f"DocumentGenerationParams({', '.join(f'{k}={v}' for k, v in self.to_dict().items() if v is not None)})"
