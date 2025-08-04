from copy import copy
from dataclasses import dataclass, field
from typing import Any, Callable

from ydata.preprocessors.exceptions import InvalidAnonymizerConfig
from ydata.preprocessors.methods.anonymization import AnonymizerType, _get_anonymizer_method


@dataclass
class ColumnAnonymizerConfiguration:
    type: AnonymizerType | str
    cols: list[str]
    params: dict[str, Any] = field(default_factory=dict)
    locale: str | None = None

    def __validate_config(self):
        if self.type == AnonymizerType.REGEX:
            if "regex" not in self.params:
                raise InvalidAnonymizerConfig(
                    f"Regex anonymizer for columns {self.cols} is missing the required parameter `regex`."
                )
        # TODO add inspect based validation

    def get_anonymizer(self):
        if self.type == "LAMBDA":
            return self.params["_function_"]
        return _get_anonymizer_method(self.type)

    def from_dict(
        key: str,
        config: dict | str | int | AnonymizerType | Callable | list,
        default_locale: str | None = None,
        validate: bool = True,
    ) -> "ColumnAnonymizerConfiguration":
        if isinstance(config, int):
            new_config = ColumnAnonymizerConfiguration(
                type=AnonymizerType(config),
                cols=[key],
                locale=default_locale,
            )
        if isinstance(config, AnonymizerType):
            new_config = ColumnAnonymizerConfiguration(
                type=config,
                cols=[key],
                locale=default_locale,
            )
        elif isinstance(config, str):
            anonymizer_type = AnonymizerType.get_anonymizer_type(
                config)
            if anonymizer_type is None:
                raise InvalidAnonymizerConfig(
                    f"Anonymizer type {config} is not available"
                )

            new_config = ColumnAnonymizerConfiguration(
                type=anonymizer_type,
                cols=[key],
                locale=default_locale,
            )
        elif isinstance(config, Callable):
            new_config = ColumnAnonymizerConfiguration(
                type="LAMBDA",
                cols=[key],
                params={"_function_": config},
                locale=default_locale,
            )
        elif isinstance(config, list):
            new_config = ColumnAnonymizerConfiguration(
                type="TEXT",
                cols=[key],
                params={
                    "text_anonymizers": [
                        ColumnAnonymizerConfiguration.from_dict(
                            key, k, default_locale=default_locale)
                        for k in config
                    ]
                },
                locale=default_locale,
            )
        elif isinstance(config, dict):
            if "type" not in config:
                raise InvalidAnonymizerConfig(
                    f"Anonymizer type for [{key}] was not defined")
            if isinstance(config["type"], dict):
                new_config = ColumnAnonymizerConfiguration.from_dict(
                    key,
                    config["type"],
                    default_locale=default_locale,
                )
            else:
                new_config = ColumnAnonymizerConfiguration.from_dict(
                    key, config["type"], validate=False, default_locale=default_locale
                )
                if "locale" in config:
                    new_config.params["locale"] = new_config.locale = config["locale"]
                if "cols" in config:
                    if isinstance(config["cols"], str):
                        new_config.cols = [config["cols"]]
                    elif isinstance(config["cols"], list):
                        new_config.cols = config["cols"]
                    else:
                        raise InvalidAnonymizerConfig(
                            f"[{key}] anonymizer [cols] type not supported."
                            f" Expected [str | list[str]] received [{type(config['cols'])}]"
                        )
                reserved_keywords = {"locale", "cols", "type"}
                for parameter, value in config.items():
                    if parameter not in reserved_keywords:
                        new_config.params[parameter] = value

        if "locale" not in new_config.params and new_config.locale:
            new_config.params["locale"] = new_config.locale
        if validate:
            new_config.__validate_config()
        return new_config

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        type_name = self.type.name if isinstance(
            self.type, AnonymizerType) else self.type
        return f"ColumnAnonymizerConfiguration(anonymizer_type={type_name}, cols={self.cols},...)"

    def to_dict(self):
        params = copy(self.params)
        if "text_anonymizers" in params:
            params["text_anonymizers"] = [
                t.to_dict()
                for t in params["text_anonymizers"]
            ]
        if "_anonymizers_" in params:
            params.pop("_anonymizers_")

        return {
            "type": self.type.name if isinstance(self.type, AnonymizerType) else self.type,
            "cols": copy(self.cols),
            "params": params,
            "locale": self.locale,
        }
