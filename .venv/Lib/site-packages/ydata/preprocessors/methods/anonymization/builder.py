from copy import copy
from dataclasses import dataclass, field
from typing import Any, Callable

from dill import dump as pdump
from dill import load as pload

from ydata.preprocessors.methods.anonymization import AnonymizerType, _get_anonymizer_method
from ydata.preprocessors.methods.anonymization.column_configuration import ColumnAnonymizerConfiguration


@dataclass
class ColumnAnonymizer:
    anonymizer: Callable
    cols: list[str]
    params: dict[str, Any] = field(default_factory=dict)


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
    VALID_METHODS = set(AnonymizerType)

    def __init__(self, initial_config=None, locale: str | None = None):
        """Initializes the ConfigurationBuilder with an optional initial
        configuration.

        Parameters:
        -----------
        initial_config : dict, optional
            An optional dictionary to initialize the configuration.
        locale : str, optional
            An optional localization for anonymizers.
        """
        self.config = dict()
        self.default_locale = locale
        if initial_config:
            self.add_config(initial_config)

    @staticmethod
    def __init_text_anonymizer(config: ColumnAnonymizerConfiguration):
        params = copy(config.params)
        if "locale" not in params:
            params["locale"] = config.locale

        if config.type == "LAMBDA":
            anonymizer_function = config.params["_function_"]
        elif config.type == "TEXT":
            anonymizer_function = _get_anonymizer_method(
                "TEXT")
        else:
            anonymizer_function = _get_anonymizer_method(
                config.type)

        return ColumnAnonymizer(
            anonymizer=anonymizer_function,
            cols=config.cols,
            params=params,
        )

    def __build_text_anonymizers(self, config: ColumnAnonymizerConfiguration):
        anonymizers = {}
        dataset_config = config.params["text_anonymizers"]

        for row_config in dataset_config:
            if row_config.type == "TEXT":
                text_anonymizer_config = config.params["text_anonymizers"]
                # get all anonymizer methods
                for method in text_anonymizer_config:
                    # has multiple anonymizers per row
                    if method.type == "TEXT":
                        for k in method.params["text_anonymizers"]:
                            name = self.__get_anonymizer_name(k)
                            if name not in anonymizers:
                                anonymizers[name] = self.__init_text_anonymizer(
                                    k)
                    # has only one anonymizers per row
                    else:
                        name = self.__get_anonymizer_name(method)
                        if name not in anonymizers:
                            anonymizers[name] = self.__init_text_anonymizer(
                                method)
            else:
                name = self.__get_anonymizer_name(row_config)
                if name not in anonymizers:
                    anonymizers[name] = self.__init_text_anonymizer(method)

        return anonymizers

    def __get_anonymizer_name(self, config: ColumnAnonymizerConfiguration) -> str:
        if isinstance(config.type, str):
            return config.type.upper()
        return config.type.name.upper()

    def add_config(
        self,
        configuration: dict,
    ):
        """Adds the anonymizer configurations.

        Args:
            config (dict): collection of column anonymizer configurations.
        """
        for key, anonymizer_def in configuration.items():
            if isinstance(anonymizer_def, ColumnAnonymizerConfiguration):
                self.config[key] = anonymizer_def
            else:
                config = ColumnAnonymizerConfiguration.from_dict(
                    key, anonymizer_def, default_locale=self.default_locale)
                if self.__get_anonymizer_name(config) == "TEXT":
                    config.params["anonymizers"] = self.__build_text_anonymizers(
                        config)
                self.config[key] = config

    def remove_config(self, column: str):
        """Removes the configuration for a specified column.

        Parameters:
        -----------
        column : str
            The name of the column whose configuration is to be removed.
        """
        if column in self.config:
            self.config.pop(column)

    def save(self, filepath: str):
        """Saves the current configuration to a file.

        Parameters:
        -----------
        filepath : str
            The path to the file where the configuration will be saved.
        """
        with open(filepath, "wb") as f:
            pdump(self, f)
            f.close()

    @staticmethod
    def load(filepath: str):
        """Loads a configuration from a file.

        Parameters:
        -----------
        filepath : str
            The path to the file from which the configuration will be loaded.
        """
        with open(filepath, "rb") as f:
            config = pload(f)

        assert isinstance(config, AnonymizerConfigurationBuilder), (
            "The loaded file must correspond to a AnonymizerConfigurationBuilder object. "
            "Please validate the given input path."
        )
        return config

    def get_config(self):
        """Returns the current configuration.

        Returns:
        --------
        dict
            The current configuration dictionary.
        """
        return self.config
