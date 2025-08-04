import datetime
from os import getenv
from string import ascii_lowercase, digits

import numpy as np
import pandas as pd
from dill import dump as pdump
from dill import load as pload
from faker import Faker

from ydata.dataset import Dataset
from ydata.metadata import Metadata
from ydata.metadata.column import Column
from ydata.synthesizers.faker.utils import (convert_characteristic_to_float, convert_characteristic_to_int,
                                            get_generator, get_n_unique_from_one, regex_generator)
from ydata.synthesizers.logger import synthlogger_config
from ydata.synthesizers.base_model import DATATYPE_MAPPING
from ydata.utils.data_types import _DATE_VARTYPES, _NUMERICAL_VARTYPES, DataType, VariableType
from ydata.utils.misc import log_time_factory
from ydata.utils.logger import SDKLogger
from ydata._licensing import synthesizer_sample


metrics_logger = SDKLogger(name="Metrics logger")

# Define here the logging status and definition
logger = synthlogger_config(verbose=getenv(
    "VERBOSE", "false").lower() == "true")


class FakerSynthesizer:
    """
    A synthesizer for generating synthetic data based on user-defined configurations.

    The `FakerSynthesizer` allows users to create synthetic tabular data **without needing
    an existing dataset**. Instead, it generates data based on user-provided **metadata**
    or manually defined column configurations. This approach is useful for:

    - Creating **mock datasets** for testing and development.
    - Generating **data prototypes** before real data is available.
    - Ensuring **privacy-preserving synthetic data** without reference to actual records.

    ## Key Features:
    - **Metadata-Driven Generation**: Generates synthetic data based on predefined Metadata.
    - **Customizable Column Types**: Supports user-defined column structures.
    - **Multi-Language Support**: Uses locale settings to generate realistic names, addresses, etc.

    ## Example Usage:
    ```python
    from ydata.synthesizers import FakerSynthesizer

    # Initialize the synthesizer with a specific locale
    faker_synth = FakerSynthesizer(locale="en")  # English data generation

    faker_synth.fit(metadata)

    # Generate synthetic data
    synthetic_data = faker_synth.sample(n_samples=1000)
    ```
    """
    __name__ = "FakerSynthesizer"

    def __init__(self, locale: str = "en") -> None:
        self.metadata = {}
        self.columns: dict[str, Column] = {}
        self.locale = locale
        self._anonymize_categoricals: bool = False

    @log_time_factory(logger)
    def fit(self, metadata: Metadata):
        """
        Configure the `FakerSynthesizer` using provided metadata.

        This method sets up the synthesizer by defining the structure of the synthetic
        dataset based on the given `Metadata`. The metadata can either be:

        - **Computed**: Automatically extracted from an existing dataset.
        - **User-Defined**: Manually constructed to specify custom column types and distributions.

        Once `fit()` is called, the synthesizer will use this metadata to generate
        structured synthetic data that adheres to the defined schema.

        Args:
            metadata: A metadata object describing the structure of the synthetic dataset, including:
                      - Column names and data types.
                      - Faker-based data generators (e.g., names, addresses, emails).
                      - Value constraints (e.g., numeric ranges, categorical options).
        """

        self.columns: dict[str, Column] = metadata.columns

        metrics_logger.info(datatype=DATATYPE_MAPPING[self.__class__.__name__],
                             method='synthesizer',
                             ncols=len(self.columns.keys()))

        self.domains: dict = metadata.summary["domains"]
        self.nunique = metadata.summary["cardinality"]
        self.missings = metadata.summary["missings"]
        self.nrows = metadata.summary["nrows"]
        self.extra_data = metadata.summary.get("extra_data", {})
        self._is_from_config = metadata.summary.get("is_from_config", False)

        if self._anonymize_categoricals:
            columns_to_mask = list(metadata.summary["value_counts"].keys())
        else:
            columns_to_mask = [
                k for k, v in self.columns.items()
                if v.characteristics
            ]
        if columns_to_mask:
            self.value_counts = self._mask_value_counts(
                metadata.summary["value_counts"],
                columns_to_mask,
            )
        else:
            self.value_counts = metadata.summary["value_counts"]

    def _is_unique(self, column: str) -> bool:
        if self._is_from_config:
            unique = self.extra_data.get(
                column, {}).get("unique", False)
            return unique
        else:
            n_valid_rows = self.nrows - self.missings[column]
            return n_valid_rows == self.nunique[column]

    @log_time_factory(logger)
    def _mask_value_counts(
        self,
        value_counts: dict[str, pd.Series],
        columns_to_mask: list[str]
    ) -> dict[str, pd.Series]:
        masked_counts = {}
        for column_name, counts in value_counts.items():
            if column_name not in columns_to_mask:
                masked_counts[column_name] = counts.copy()
                continue

            column = self.columns[column_name]
            nunique = len(counts)
            params = {
                "column": column,
                "sample_size": nunique,
                "unique": True
            }
            if column.vartype == VariableType.STR:
                new_index = self._sample_string_values(**params)
            elif column.vartype in _NUMERICAL_VARTYPES:
                new_index = self._sample_numerical_values(**params)
            elif column.vartype in _DATE_VARTYPES:
                new_index = self._sample_date_values(**params)
            else:
                # there is no need to change booleans
                new_index = counts.index
            masked_counts[column_name] = counts.copy()
            masked_counts[column_name].index = new_index
        return masked_counts

    def _get_locale(self, column: str):
        return self.extra_data.get(column, {}).get("locale", self.locale)


    def _MultiTableSynthesizer__sample(self, sample_size=1000) -> Dataset:
        """Auxiliary function to avoid having Multitable charging twice the users"""
        if self._is_from_config:
            self.nrows = sample_size
            for col in self.columns:
                if col not in self.nunique:
                    self.nunique[col] = sample_size

        series: list[pd.Series] = []
        for col in self.columns.values():
            data = self._sample_column(col, sample_size)
            series.append(pd.Series(data, name=col.name))
        if series:
            df = pd.concat(series, axis=1)
        else:
            df = pd.DataFrame()
        return Dataset(df)

    @log_time_factory(logger)
    @synthesizer_sample
    def sample(self, sample_size=1000) -> Dataset:
        """
        Generate a synthetic dataset based on the configured metadata.

        This method produces synthetic data according to the schema defined in the
        `fit()` step. The generated data adheres to the column types, constraints,
        and distributions specified in the provided `Metadata`.

        Args:
            sample_size (int): The number of synthetic records/rows to generate. Defaults to `1000`.

        Returns:
            dataset (Dataset): A Dataset object with the generated synthetic records/rows.

        """
        return self._MultiTableSynthesizer__sample(sample_size=sample_size)

    def _sample_characteristic(self, column: Column, sample_size: int, characteristic, unique: bool) -> list:
        locale = self._get_locale(column.name)
        gen = get_generator(characteristic, locale)
        data = get_n_unique_from_one(
            gen,
            sample_size,
            unique=unique
        )
        # TODO list all type/characteristic combinations
        # TODO add unit tests
        if column.vartype == VariableType.INT:
            data = convert_characteristic_to_int(data)
        if column.vartype == VariableType.FLOAT:
            data = convert_characteristic_to_float(data)
        return data

    def _sample_column(self, column: Column, sample_size: int) -> list:
        n_missings = self._get_expected_number_missing_values(
            column, sample_size)
        params = {
            "column": column,
            "sample_size": sample_size - n_missings,
            "unique": self._is_unique(column.name),
        }
        if n_missings >= sample_size:
            return self._get_missing_values(column, sample_size)
        if column.vartype in _NUMERICAL_VARTYPES:
            data = self._sample_numericals(**params)
        elif column.vartype in _DATE_VARTYPES:
            data = self._sample_dates(**params)
        elif column.vartype == VariableType.STR:
            data = self._sample_strings(**params)
        elif column.vartype == VariableType.BOOL:
            data = self._sample_booleans(
                column=column,
                sample_size=sample_size,
            )
            return data
        else:
            raise ValueError(
                f"Unkown variable type [{column.vartype}] for column [{column.name}].")

        data += self._get_missing_values(column, n_missings)
        np.random.shuffle(data)
        return data

    def _get_missing_values(self, column: Column, n_missings: int) -> list:
        missing_type = None
        if column.vartype in [VariableType.INT, VariableType.FLOAT]:
            missing_type = np.nan
        elif column.vartype in [VariableType.DATE, VariableType.DATETIME]:
            missing_type = pd.NaT
        return [missing_type] * n_missings

    def _get_expected_number_missing_values(self, column: Column, sample_size: int) -> int:
        n_missing = self.missings.get(column.name, 0)
        if self._is_from_config:
            missing_probability = n_missing / 100
        else:
            missing_probability = n_missing / self.nrows
        return int(np.floor(missing_probability * sample_size))

    def _sample_numericals(
        self,
        column: Column,
        sample_size: int,
        unique: bool
    ) -> list:
        if column.datatype == DataType.CATEGORICAL:
            return self._sample_categorical(
                column=column,
                sample_size=sample_size,
                unique=unique,
            )
        elif column.datatype == DataType.NUMERICAL:
            return self._sample_numerical_values(
                column=column,
                sample_size=sample_size,
                unique=unique
            )
        else:
            raise ValueError(
                f"Unsupported data type {column.datatype.value} for int/float vartype in [{column.name}] column.")

    def _sample_numerical_values(self, column: Column, sample_size: int, unique: bool = False) -> list:
        if column.characteristics:
            data = self._sample_characteristic(
                column=column,
                sample_size=sample_size,
                unique=unique,
                characteristic=column.characteristics[0]
            )
            return data

        # get domain
        domain = self.domains.get(column.name, {})
        assert "max" in domain, f"max not defined for column {column.name}."
        assert "min" in domain, f"min not defined for column {column.name}."
        if column.vartype == VariableType.INT:
            domain["min"] = int(domain["min"])
            domain["max"] = int(domain["max"])
        else:
            domain["min"] = float(domain["min"])
            domain["max"] = float(domain["max"])
        error_msg = f"Invalid range of values, max < min for column {column.name}."
        assert domain["max"] >= domain["min"], error_msg

        if column.vartype == VariableType.INT:
            domain["min"] = int(domain["min"])
            domain["max"] = int(domain["max"])
            domain_size = domain["max"] - domain["min"] + 1
            if unique and domain_size == sample_size:
                data = np.arange(domain["min"], domain["max"] + 1)
            elif unique and domain_size > sample_size:
                data = np.arange(
                    domain["min"],
                    domain["min"] + sample_size
                )
                available_values = domain_size - sample_size + 1
                offset = np.random.randint(available_values, size=sample_size)
                offset.sort()
                data += offset
            else:  # not unique or (unique and domain_size < sample_size)
                if unique and domain_size < sample_size:
                    logger.warning(
                        f"Not enough unique values to generate for column {column.name}.")
                data = np.random.randint(
                    low=domain["min"], high=domain["max"] + 1, size=sample_size)
        # VariableType.FLOAT
        else:
            if not unique:
                data = np.random.uniform(
                    low=domain["min"], high=domain["max"], size=sample_size)
            else:
                data = np.random.uniform(
                    low=domain["min"], high=domain["max"], size=sample_size)
                data = set(data)
                while len(data) < sample_size:
                    data.add(
                        np.random.uniform(
                            low=domain["min"], high=domain["max"]
                        )
                    )
                data = list(data)

        return data.tolist() if isinstance(data, np.ndarray) else data

    def _sample_dates(self, column: Column, sample_size: int, unique: bool) -> list:
        if column.datatype == DataType.CATEGORICAL:
            return self._sample_categorical(
                column=column,
                sample_size=sample_size,
                unique=unique,
            )
        elif column.datatype == DataType.DATE:
            return self._sample_date_values(
                column=column,
                sample_size=sample_size,
                unique=unique,
            )
        else:
            raise ValueError(
                f"Unsupported data type {column.datatype.value} for date/datetime vartype in [{column.name}] column.")

    def _sample_date_values(self, column: Column, sample_size: int, unique: bool = False) -> list:
        domain = self.domains.get(column.name, {})
        date_format = self.extra_data.get(column.name, {}).get("format")
        if "min" in domain and isinstance(domain["min"], str):
            domain["min"] = datetime.datetime.strptime(
                domain["min"], date_format)
        if "max" in domain and isinstance(domain["max"], str):
            domain["max"] = datetime.datetime.strptime(
                domain["max"], date_format)
        if "max" not in domain:
            domain["max"] = datetime.datetime.now()
        if "min" not in domain:
            domain["min"] = domain["max"] - datetime.timedelta(days=365 * 10)

        # is constant?
        if self.nunique[column.name] == 1:
            return [domain["min"]] * sample_size

        diretives_mapping = [
            ("microseconds", ["%f"]),
            ("seconds", ["%S", "%c", "%X"]),
            ("minutes", ["%M"]),
            ("hours", ["%H", "%I", "%p"]),
            ("days", ["%a", "%A", "%w", "%d", "%j", "%x", "%u"]),
            ("weeks", ["%U", "%W", "%V"]),
            ("months", ["%b", "%B", "%m"]),
            ("years", ["%y", "%Y", "%G"]),
        ]

        delta_magnitude = "microseconds"
        if self._is_from_config and date_format:
            for magnitude, directives in diretives_mapping:
                if any([d in date_format for d in directives]):
                    delta_magnitude = magnitude
                    break

        magnitude_offset = 1
        if delta_magnitude == "years":
            magnitude_offset = 365
            delta_magnitude = "days"
        elif delta_magnitude == "months":
            magnitude_offset = 31
            delta_magnitude = "days"

        domain_size = domain["max"] - domain["min"]
        domain_size = domain_size / datetime.timedelta(**{delta_magnitude: 1})
        domain_size += magnitude_offset  # to includes the max value
        if unique and domain_size == sample_size:
            data = [
                domain["min"] +
                datetime.timedelta(**{delta_magnitude: float(t)})
                for t in np.arange(domain_size)
            ]
        elif unique and domain_size > sample_size:
            delta = np.random.randint(domain_size, size=sample_size)
            data = [
                domain["min"] +
                datetime.timedelta(**{delta_magnitude: float(t)})
                for t in np.arange(sample_size)
            ]
            extra_values = domain_size - sample_size + 1
            offset = np.random.randint(extra_values, size=sample_size)
            offset.sort()
            for i, value in enumerate(data):
                data[i] = value + \
                    datetime.timedelta(**{delta_magnitude: float(offset[i])})
        else:  # not unique or (unique and domain_size < sample_size)
            if unique and domain_size < sample_size:
                logger.warning(
                    f"Not enough unique values to generate for column {column.name}.")
            delta = np.random.randint(domain_size, size=sample_size)
            data = [
                domain["min"] +
                datetime.timedelta(**{delta_magnitude: float(t)})
                for t in delta
            ]
        return data

    def _sample_strings(self, column: Column, sample_size: int, unique: bool) -> list[str]:
        if column.datatype == DataType.CATEGORICAL:
            return self._sample_categorical(
                column=column,
                sample_size=sample_size,
                unique=unique,
            )
        elif column.datatype in [DataType.LONGTEXT, DataType.STR]:
            return self._sample_string_values(
                column=column,
                sample_size=sample_size,
                unique=unique,
            )
        else:
            raise ValueError(
                f"Unsupported data type {column.datatype.value} for string vartype in [{column.name}] column.")

    def _sample_random_string_values(self, sample_size: int, unique: bool = False) -> list[str]:
        valid_characters = list(ascii_lowercase + digits)

        def generate_string(size: int) -> str:
            return "".join(np.random.choice(valid_characters, size=size))

        minimun_lenth = (sample_size // len(valid_characters)) + 1
        data = [
            generate_string(minimun_lenth)
            for _ in range(sample_size)
        ]

        if unique:
            data = set(data)
            while len(data) < sample_size:
                length = np.random.randint(
                    minimun_lenth + 1, minimun_lenth + 10)
                data.add(generate_string(length))
            data = list(data)

        return data

    def _sample_string_values(self, column: Column, sample_size: int, unique: bool = False) -> list[str]:
        regex = self.extra_data.get(column.name, {}).get("regex")
        if regex:
            data = self._sample_from_regex(
                column=column,
                sample_size=sample_size,
                unique=unique
            )
            return data
        if column.characteristics:
            data = self._sample_characteristic(
                column=column,
                sample_size=sample_size,
                characteristic=column.characteristics[0],
                unique=unique,
            )
            return data
        text_datatypes = [
            DataType.STR,
            DataType.CATEGORICAL,
            DataType.LONGTEXT
        ]
        if column.datatype in text_datatypes:
            data = self._sample_text_values(
                column=column,
                sample_size=sample_size,
                unique=unique
            )
            return data

        raise ValueError(
            f"Unsupported data type {column.datatype.value} for string vartype in [{column.name}] column.")

    def _sample_text_values(self, column: Column, sample_size: int, unique: bool = False) -> list[str]:
        fake = Faker(locale=self._get_locale(column.name))
        text_length = self.extra_data.get(
            column.name, {}
        ).get("text_length", 50)
        if column.datatype == DataType.STR:
            text_length = 50
        data = fake.texts(
            nb_texts=sample_size,
            max_nb_chars=text_length
        )
        if unique:
            data = set(data)
            MAX_ITERATIONS = 10
            iteration = 0
            while len(data) < sample_size and iteration < MAX_ITERATIONS:
                new_data = fake.texts(
                    nb_texts=sample_size - len(data),
                    max_nb_chars=text_length
                )
                data |= set(new_data)
                iteration += 1

            data = list(data)
            if len(data) < sample_size:
                logger.warning(
                    f"Unable to generate enough unique values for column `{column.name}`. Allowing duplicated values")
                new_data = fake.texts(
                    nb_texts=sample_size - len(data),
                    max_nb_chars=text_length
                )
                data.extend(new_data)

        return data

    def _sample_booleans(self, column: Column, sample_size: int) -> list:
        """Sample random boolean values.

        Args:
            column (Column): column to sample
            sample_size (int): amount of values to sample

        Returns:
            list: random boolean values
        """
        return self._sample_categorical(
            column=column,
            sample_size=sample_size,
            unique=False,
        )

    def _sample_from_regex(self, column: str, sample_size: int, unique: bool) -> list:
        regex = self.extra_data[column.name]["regex"]
        values = get_n_unique_from_one(
            regex_generator(regex),
            sample_size,
            unique=unique
        )
        return values

    def _get_categorical_distribution(self, column: Column) -> tuple[list, list]:
        if column.name in self.value_counts:
            dist = self.value_counts[column.name] / \
                self.value_counts[column.name].sum()
            return dist.index.to_list(), dist.values
        else:
            nunique = self.nunique[column.name]
            params = {
                "column": column,
                "sample_size": nunique,
                "unique": True
            }
            if "regex" in self.extra_data.get(column.name, {}):
                values = self._sample_from_regex(**params)
            elif column.vartype in _NUMERICAL_VARTYPES:
                values = self._sample_numerical_values(**params)
            elif column.vartype in _DATE_VARTYPES:
                values = self._sample_date_values(**params)
            elif column.vartype == VariableType.STR:
                values = self._sample_string_values(**params)
            else:
                values = [True, False]

            nunique = len(values)
            prob = [1 / nunique] * nunique
            return values, prob

    def _sample_categorical(self, column: Column, sample_size: int, unique: bool) -> list:
        values, probability = self._get_categorical_distribution(column)
        if unique and sample_size <= self.nunique[column.name]:
            data = np.random.choice(
                values,
                p=probability,
                size=sample_size,
                replace=False
            ).tolist()
        else:  # if not unique or (unique and sample_size > nunique)
            if unique and sample_size > self.nunique[column.name]:
                logger.warning(
                    f"Not enough unique values to generate for column {column.name}. Allowing duplicates")
            data = []
            for val, prob in zip(values, probability):
                data += [val] * int((np.floor(sample_size * prob)))
            data += np.random.choice(
                values,
                p=probability,
                size=sample_size - len(data),
                replace=True
            ).tolist()

            if column.vartype in _DATE_VARTYPES and isinstance(values[0], str):
                date_format = self.extra_data.get(
                    column.name, {}).get('format')
                if date_format is not None:
                    data = [datetime.datetime.strptime(
                        d, date_format) for d in data]
                else:
                    data = [datetime.datetime.strptime(d) for d in data]
        return data

    def save(self, path):
        """Saves the SYNTHESIZER and the model fitted per variable."""

        logger.info("[SYNTHESIZER] - Saving SYNTHESIZER state.")

        with open(path, "wb") as f:
            pdump(self, f)
            f.close()

    @classmethod
    def load(cls, path: str):
        logger.info("[SYNTHESIZER] - Loading SYNTHESIZER state.")
        with open(path, "rb") as f:
            synth = pload(f)

        assert isinstance(synth, FakerSynthesizer), (
            "The loaded file must correspond to a FakerSynthesizer object. "
            "Please validate the given input path."
        )

        return synth
