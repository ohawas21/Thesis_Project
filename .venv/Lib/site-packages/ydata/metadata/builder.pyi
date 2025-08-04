from _typeshed import Incomplete
from typing import Any

logger: Incomplete
TRUTHY_STR_VALUES: Incomplete
FALSY_STR_VALUES: Incomplete
ERROR_MARGIN: float

class MetadataConfigurationBuilder:
    """Builder class for creating and managing metadata configurations.

    Methods:
        __init__(config_dict=None): Initializes the builder with an optional configuration dictionary.
        add_new_column(name, datatype, vartype, **options): Adds a new column to the metadata.
        remove_column(name): Removes a column from the metadata by name.
        _validate_input(): Validates the provided configuration dictionary.
        load(file_path): Loads metadata configurations from a pickle or YAML file.
        save(file_path): Saves the metadata configurations to a pickle or YAML file.
        build(): Builds and returns a list of Metadata objects based on the current configurations.
    """
    config: Incomplete
    def __init__(self, config: dict[str, Any] = None) -> None:
        """Validates the provided inputs and set the provided config of the
        builder."""
    def add_column(self, column: str, datatype: str, vartype: str, characteristic: str | None = None, **options):
        """Adds a new column to the metadata.

        Args:
            column (str): The name of the column.
            datatype (str): The type of data (categorical, long text, numerical, date).
            vartype (str): The variable type (int, float, string).
            characteristic (str | None): Optional column characteristic (e.g email, name)
            **options: Additional options for data generation.
        """
    def remove_column(self, column: str):
        """Removes a column from the metadata by column.

        Args:
            column (str): The column of the column to remove.
        """
    @staticmethod
    def load(file_path: str) -> MetadataConfigurationBuilder:
        """Loads metadata configurations from a pickle or YAML file.

        Args:
            file_path (str): The path to the file to load.
        """
    def save(self, file_path: str):
        """Saves the metadata configurations to a pickle or YAML file.

        Args:
            file_path (str): The path to the file to save.
        """
