import re
from datetime import datetime
from os import getenv
from typing import Any

from dill import dump as pdump
from dill import load as pload

from ydata.characteristics import ColumnCharacteristic
from ydata.metadata.utils import is_non_negative_number
from ydata.utils.data_types import DATA_VARTYPE_MAP, DataType, VariableType, is_characteristic_type_valid
from ydata.utils.logger import utilslogger_config

logger = utilslogger_config(verbose=getenv(
    "VERBOSE", "false").lower() == "true")


TRUTHY_STR_VALUES = {"true", "t", "y", "1"}
FALSY_STR_VALUES = {"false", "f", "n", "0"}
ERROR_MARGIN = 1e-6


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

    def __init__(self, config: dict[str, Any] = None):
        """Validates the provided inputs and set the provided config of the
        builder."""
        self.config = {}

        if config:
            self._validate_input(config)
            for column, params in config.items():
                self.add_column(column, **params)

    @staticmethod
    def __is_parameter_type_valid(parameter, enum: DataType | VariableType | ColumnCharacteristic):
        if isinstance(parameter, enum.__class__):
            return True
        elif isinstance(parameter, str):
            return (
                parameter.upper() in enum._member_names_
                or parameter in enum._value2member_map_
            )
        else:
            return parameter in enum._value2member_map_

    def _validate_characteristic(self, column: str, vartype: str, datatype: str, characteristic: str):
        if not self.__is_parameter_type_valid(characteristic, ColumnCharacteristic):
            raise ValueError(
                f"The characteristic {characteristic} is not a valid ColumnCharacteristic")

        characteristic = ColumnCharacteristic(characteristic)
        invalid_characteristics = {
            ColumnCharacteristic.PII,
            ColumnCharacteristic.LOCATION,
            ColumnCharacteristic.PERSON
        }
        if characteristic in invalid_characteristics:
            raise ValueError(
                f"characteristic {characteristic.value} is not a supported PII type for generating synthetic data with FakerSynthesizer. Please provide a different PII type.")

        if not is_characteristic_type_valid(characteristic, VariableType(vartype)):
            raise ValueError(
                f"characteristic {characteristic} is not supported for vartype {vartype} in column {column}.")

        if DataType(datatype.lower()) == DataType.LONGTEXT:
            raise ValueError(
                f"{DataType.LONGTEXT.value} columns does not supported PIIs generation.")

    def _validate_column_options(self, column: str, datatype: str, vartype: str, characteristic: str = None, **options):
        """Validate the options selected to define the column.

        Args:
            column (str): column name
            datatype (str): column data type
            vartype (str): column variable type
            characteristic (str, optional): column characteristic. Defaults to None.
            options (dict): column build parameters

        Raises:
            ValueError: validation error
        """
        vartype = VariableType(vartype.lower())
        datatype = DataType(datatype.lower())
        if vartype in DATA_VARTYPE_MAP[DataType.DATE]:
            if datatype == DataType.DATE:
                if "min" not in options or "max" not in options:
                    raise ValueError(
                        f"Date min/max not defined for column {column}.")

                # string dates defined without format
                dates = [options["min"], options["max"]]
                if any([isinstance(d, str) for d in dates]) and not options.get("format"):
                    raise ValueError(
                        f"Date format not defined for column {column}.")
                if isinstance(options["min"], str):
                    min_ = datetime.strptime(options["min"], options["format"])
                else:
                    min_ = options["min"]
                if isinstance(options["max"], str):
                    max_ = datetime.strptime(options["max"], options["format"])
                else:
                    max_ = options["max"]

                if min_ > max_:
                    raise ValueError(
                        f"Invalid range of values, max < min for column {column}.")
            elif datatype == DataType.CATEGORICAL:
                categories = options.get("categories", {})
                if categories and isinstance(categories, dict):
                    dates = categories.keys()
                    # string dates defined without format
                    if any([isinstance(d, str) for d in dates]) and not options.get("format"):
                        raise ValueError(
                            f"Date format not defined for column {column}.")

        # validate regex
        if vartype == VariableType.STR and "regex" in options:
            try:
                re.compile(options["regex"])
            except re.error:
                raise ValueError(
                    f"regex {options['regex']} for column {column} is invalid")

        if datatype == DataType.CATEGORICAL:
            if "categories" not in options:
                raise ValueError(
                    f"categories is required for {DataType.CATEGORICAL.value}.")
            elif not (isinstance(options["categories"], dict) or isinstance(options["categories"], list)):
                raise ValueError(
                    "Ensure categories type is a list or dictionary.")
            elif len(options["categories"]) == 0:
                raise ValueError(
                    "Please review your inputs for the listed categories. Ensure categories is not empty.")
            else:
                if isinstance(options["categories"], dict):
                    probabilities = list(options["categories"].values())
                else:
                    probabilities = options["categories"]
                if not all([is_non_negative_number(p) for p in probabilities]):
                    raise ValueError(
                        "Please review your inputs for the listed categories. Ensure that all probabilities are positive and sum to 100%.")
                total_prob = sum(probabilities)
                if total_prob < 100 - ERROR_MARGIN or total_prob > 100 + ERROR_MARGIN:
                    raise ValueError(
                        "Please review your inputs for the listed categories. Ensure that the total sum to 100%.")

        if datatype == DataType.NUMERICAL:
            if not characteristic:
                if "max" not in options or "min" not in options:
                    raise ValueError(
                        f"min/max were not defined for column {column}.")
                if vartype == VariableType.INT:
                    min_ = int(options["min"])
                    max_ = int(options["max"])
                else:
                    min_ = float(options["min"])
                    max_ = float(options["max"])
                if min_ > max_:
                    raise ValueError(
                        f"Invalid range of values, max < min for column {column}.")

        # is categorical bool?
        if vartype == VariableType.BOOL:
            categories = options["categories"]
            if len(categories) > 2:
                raise ValueError(
                    "Boolean columns can have at most 2 categories.")
            bools = {True, False}
            if any([cat not in bools for cat in categories]):
                raise ValueError(
                    "Invalid boolean values as categories.")
            if "missings" in options:
                raise ValueError(
                    "Boolean values do not allow missing values.")

        if datatype == DataType.LONGTEXT:
            if "text_length" not in options:
                raise ValueError(
                    f"Please provide the expected number of characters for the [{column}] text to be generated."
                )
            elif not isinstance(options["text_length"], int):
                raise ValueError(
                    f"text_length type for [{column}] is invalid, a int is expected."
                )

        if "missings" in options:
            missings = options["missings"]
            if not is_non_negative_number(missings) or missings > 100:
                raise ValueError(
                    f"missings probability for [{column}] is invalid."
                )

        return True

    def _validate_column_type_combinations(self, datatype: str, vartype: str) -> bool:
        # validate if datatype and vartype exists
        if not self.__is_parameter_type_valid(datatype, DataType):
            raise ValueError(
                f"The datatype {datatype} is not a valid DataType")
        if not self.__is_parameter_type_valid(vartype, VariableType):
            raise ValueError(
                f"The vartype {vartype} is not a valid VariableType")
        # validate if the vartype and datatype are valid
        if VariableType(vartype) not in DATA_VARTYPE_MAP[DataType(datatype.lower())]:
            raise ValueError(
                f"The vartype {vartype} is not a valid for datatype {datatype}")

    def _validate_column_configuration(
        self,
        column: str,
        datatype: str,
        vartype: str,
        characteristic: str | None = None,
        **options
    ):
        self._validate_column_type_combinations(
            datatype=datatype, vartype=vartype)
        if characteristic:
            self._validate_characteristic(
                column, vartype, datatype, characteristic)
        self._validate_column_options(
            column, datatype, vartype, characteristic, **options)

    def _preprocess_boolean_categoricals(self, vartype: str, **parameters) -> dict:
        if VariableType(vartype.lower()) == VariableType.BOOL:
            categories = parameters.get("categories", {})
            if not isinstance(categories, dict):
                return parameters
            elif len(categories) < 1 or len(categories) > 2:
                return parameters
            else:
                new_categories = {}
                for cat, prob in categories.items():
                    if str(cat).strip().lower() in TRUTHY_STR_VALUES:
                        if True not in new_categories:
                            new_categories[True] = prob
                        else:
                            new_categories[cat] = prob
                    elif str(cat).strip().lower() in FALSY_STR_VALUES:
                        if False not in new_categories:
                            new_categories[False] = prob
                        else:
                            new_categories[cat] = prob
                if new_categories:
                    parameters["categories"] = new_categories
        return parameters

    def _preprocess_categorical_characteristics(
        self,
        datatype: str,

        characteristic: str | None = None,
        **parameters
    ) -> dict:
        if characteristic and DataType(datatype.lower()) == DataType.CATEGORICAL:
            if "categories" in parameters:
                categories = parameters["categories"]
                # if categories contains only the prob distribution
                if isinstance(categories, list):
                    # create the categories map
                    categories = {
                        f"{characteristic}_{i}": prob
                        for i, prob in enumerate(categories)
                    }
                    parameters["categories"] = categories
        return parameters

    def _preprocess_parameters(
        self,
        datatype: str,
        vartype: str,
        characteristic: str | None = None,
        **parameters
    ) -> dict:
        parameters = self._preprocess_boolean_categoricals(
            vartype, **parameters)
        parameters = self._preprocess_categorical_characteristics(
            datatype, characteristic, **parameters)
        return parameters

    def add_column(self, column: str, datatype: str, vartype: str, characteristic: str | None = None, **options):
        """Adds a new column to the metadata.

        Args:
            column (str): The name of the column.
            datatype (str): The type of data (categorical, long text, numerical, date).
            vartype (str): The variable type (int, float, string).
            characteristic (str | None): Optional column characteristic (e.g email, name)
            **options: Additional options for data generation.
        """
        if column in self.config:
            raise ValueError(
                f"The dataset to be created can't have 2 columns with the same name. Please revisit your inputs for the {column}."
            )
        options = self._preprocess_parameters(
            datatype, vartype, characteristic, **options)
        self._validate_column_configuration(
            column, datatype, vartype, characteristic, **options)

        self.config[column] = {
            "datatype": DataType(datatype.lower()),
            "vartype": VariableType(vartype.lower()),
            **options
        }
        if characteristic:
            self.config[column]["characteristic"] = ColumnCharacteristic(
                characteristic.lower())

    def remove_column(self, column: str):
        """Removes a column from the metadata by column.

        Args:
            column (str): The column of the column to remove.
        """
        if column in self.config:
            self.config.pop(column)

    def _validate_input(self, config_dict: dict[str, Any]):
        """Validates the provided configuration dictionary.

        Args:
            config_dict (dict): The configuration dictionary to validate.
        """
        msg = "%s parameter is missing in the %s column configuration"
        for col, config in config_dict.items():
            assert "datatype" in config, msg % ("datatype", col)
            assert "vartype" in config, msg % ("vartype", col)

    @staticmethod
    def load(file_path: str) -> "MetadataConfigurationBuilder":
        """Loads metadata configurations from a pickle or YAML file.

        Args:
            file_path (str): The path to the file to load.
        """
        with open(file_path, "rb") as f:
            builder = pload(f)

        assert isinstance(builder, MetadataConfigurationBuilder), (
            "The loaded file must correspond to a MetadataConfigurationBuilder object. "
            "Please validate the given input path."
        )
        return builder

    def save(self, file_path: str):
        """Saves the metadata configurations to a pickle or YAML file.

        Args:
            file_path (str): The path to the file to save.
        """
        with open(file_path, "wb") as f:
            pdump(self, f)
            f.close()
