from _typeshed import Incomplete
from ydata.metadata import Metadata
from ydata.metadata.multimetadata import MultiMetadata

CHARAC_TO_ANONYM: Incomplete
CHARAC_GROUPS: Incomplete

def suggest_anonymizer_config(metadata: Metadata | MultiMetadata) -> dict[str, list[str] | dict[str, list[str]]]:
    """Suggest anonymizers based on the characteristics.

    Args:
        metadata (Metadata | MultiMetadata): dataset's metadata

    Returns:
        Dictionary mapping columns to a list of anonymizers
    """
def deduce_anonymizer_config_for_STR(metadata: Metadata) -> dict[str, str]:
    """Deduce how to generate DataType.STR using the anonymizer.

    Args:
        metadata (Metadata): dataset's metadata

    Returns:
        Dictionary mapping columns to an anonymizer
    """
def deduce_anonymizer_config_for_PII(metadata: Metadata) -> dict[str, str]:
    """Deduce how to generate DataType.STR using the anonymizer.

    Args:
        metadata (Metadata): dataset's metadata

    Returns:
        Dictionary mapping columns to an anonymizer
    """
