from _typeshed import Incomplete
from dataclasses import dataclass
from typing import Any, Callable

@dataclass
class ColumnAnonymizer:
    anonymizer: Callable
    cols: list[str]
    params: dict[str, Any] = ...
    def __init__(self, anonymizer, cols, params=...) -> None: ...

class AnonymizerConfigurationBuilder:
    """A class to build and manage configurations for data anonymization.

    Attributes:
    -----------
    VALID_METHODS : set
        A set of valid anonymization methods.

    Methods:
    --------
    __init__(initial_config=None):
        Initializes the ConfigurationBuilder with an optional initial configuration.

    validate_and_set_config(config):
        Validates and sets the initial configuration dictionary.

    add_new_config(column, method):
        Adds a new configuration for a specified column and anonymization method.

    remove_config(column):
        Removes the configuration for a specified column.

    save(filepath):
        Saves the current configuration to a file.

    load(filepath):
        Loads a configuration from a file.

    get_config():
        Returns the current configuration.
    """
    VALID_METHODS: Incomplete
    config: Incomplete
    default_locale: Incomplete
    def __init__(self, initial_config: Incomplete | None = None, locale: str | None = None) -> None:
        """Initializes the ConfigurationBuilder with an optional initial
        configuration.

        Parameters:
        -----------
        initial_config : dict, optional
            An optional dictionary to initialize the configuration.
        locale : str, optional
            An optional localization for anonymizers.
        """
    def add_config(self, configuration: dict):
        """Adds the anonymizer configurations.

        Args:
            config (dict): collection of column anonymizer configurations.
        """
    def remove_config(self, column: str):
        """Removes the configuration for a specified column.

        Parameters:
        -----------
        column : str
            The name of the column whose configuration is to be removed.
        """
    def save(self, filepath: str):
        """Saves the current configuration to a file.

        Parameters:
        -----------
        filepath : str
            The path to the file where the configuration will be saved.
        """
    @staticmethod
    def load(filepath: str):
        """Loads a configuration from a file.

        Parameters:
        -----------
        filepath : str
            The path to the file from which the configuration will be loaded.
        """
    def get_config(self):
        """Returns the current configuration.

        Returns:
        --------
        dict
            The current configuration dictionary.
        """
