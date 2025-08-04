from typing import Literal, Optional, Tuple

from dask.array import array as ddarray
from dask.dataframe import DataFrame as ddDataframe
from faker import Faker
from numpy import nan
from pandas import NA
from pandas import DataFrame as pdDataFrame

from ydata.preprocessors.methods.anonymization.utils import (get_n_unique_from_one, get_one_regex, random_ipv4,
                                                             random_ipv6)


def anonymizer_from_one(gen, X: ddDataframe, cardinality: dict, unique: bool = True):
    X_ = X.melt()['value'].dropna()
    K = sum(cardinality.values())
    vals = get_n_unique_from_one(
        gen=gen, n=K, unique=unique)
    mapping = dict(zip(X_.unique(), vals))
    for k in X.columns:
        X[k] = X[k].map(mapping)
        if X[k].dtype == "category":
            X[k] = X[k].fillna(nan)
        else:
            X[k] = X[k].replace({NA: nan})
    return X


def regex_anonymizer(X: ddDataframe, cardinality: dict, regex: str = "", unique: bool = True, **kwargs) -> str:
    return anonymizer_from_one(get_one_regex(regex), X, cardinality, unique)


def ip_anonymizer(X: ddDataframe, cardinality: dict, unique: bool = True, **kwargs) -> str:
    return ipv4_anonymizer(X, cardinality, unique)


def ipv4_anonymizer(X: ddDataframe, cardinality: dict, unique: bool = True, **kwargs) -> str:
    return anonymizer_from_one(random_ipv4, X, cardinality, unique)


def ipv6_anonymizer(X: ddDataframe, cardinality: dict, unique: bool = True, **kwargs) -> str:
    return anonymizer_from_one(random_ipv6, X, cardinality, unique)


def hostname_anonymizer(X: ddDataframe, cardinality: dict, levels: Literal[1, 2, 3] = 1, locale: str = "en", unique: bool = True, **kwargs) -> str:
    fake = Faker(locale=locale)
    return anonymizer_from_one(lambda: fake.hostname(levels), X, cardinality, unique)


def license_plate_anonymizer(X: ddDataframe, cardinality: dict, locale: str = "en", unique: bool = True, **kwargs) -> str:
    fake = Faker(locale=locale)
    return anonymizer_from_one(fake.license_plate, X, cardinality, unique)


def aba_anonymizer(X: ddDataframe, cardinality: dict, locale: str = "en", unique: bool = True, **kwargs) -> str:
    # https://en.wikipedia.org/wiki/ABA_routing_transit_number
    fake = Faker(locale=locale)
    return anonymizer_from_one(fake.aba, X, cardinality, unique)


def bank_country_anonymizer(X: ddDataframe, cardinality: dict, locale: str = "en", unique: bool = True, **kwargs) -> str:
    fake = Faker(locale=locale)
    return anonymizer_from_one(fake.bank_country, X, cardinality, unique)


def bban_anonymizer(X: ddDataframe, cardinality: dict, locale: str = "en", unique: bool = True, **kwargs) -> str:
    fake = Faker(locale=locale)
    return anonymizer_from_one(fake.bban, X, cardinality, unique)


def iban_anonymizer(X: ddDataframe, cardinality: dict, locale: str = "en", unique: bool = True, **kwargs) -> str:
    fake = Faker(locale=locale)
    return anonymizer_from_one(fake.iban, X, cardinality, unique)


def postcode_anonymizer(X: ddDataframe, cardinality: dict, locale: str = "en", unique: bool = True, **kwargs) -> str:
    fake = Faker(locale=locale)
    return anonymizer_from_one(fake.postcode, X, cardinality, unique)


def swift_anonymizer(
    X: ddDataframe, cardinality: dict, length: int = 11, primary: bool = False, use_dataset: bool = False, locale: str = "en", unique: bool = True, **kwargs
) -> str:
    fake = Faker(locale=locale)
    return anonymizer_from_one(lambda: fake.swift(length, primary, use_dataset), X, cardinality, unique)


def barcode_anonymizer(X: ddDataframe, cardinality: dict, length: int = 13, prefixes: Tuple = (), locale: str = "en", unique: bool = True, **kwargs) -> str:
    fake = Faker(locale=locale)
    return anonymizer_from_one(lambda: fake.ean(length, prefixes), X, cardinality, unique)


def company_anonymizer(X: ddDataframe, cardinality: dict, locale: str = "en", unique: bool = True, **kwargs) -> str:
    fake = Faker(locale=locale)
    return anonymizer_from_one(fake.company, X, cardinality, unique)


def company_suffix_anonymizer(X: ddDataframe, cardinality: dict, locale: str = "en", unique: bool = True, **kwargs) -> str:
    fake = Faker(locale=locale)
    return anonymizer_from_one(fake.company_suffix, X, cardinality, unique)


def company_email_anonymizer(X: ddDataframe, cardinality: dict, locale: str = "en", unique: bool = True, **kwargs) -> str:
    fake = Faker(locale=locale)
    return anonymizer_from_one(fake.company_email, X, cardinality, unique)


def email_anonymizer(X: ddDataframe, cardinality: dict, locale: str = "en", unique: bool = True, **kwargs) -> str:
    fake = Faker(locale=locale)
    return anonymizer_from_one(fake.email, X, cardinality, unique)


def domain_name_anonymizer(X: ddDataframe, cardinality: dict, levels: Literal[1, 2, 3] = 1, locale: str = "en", unique: bool = True, **kwargs) -> str:
    fake = Faker(locale=locale)
    return anonymizer_from_one(lambda: fake.domain_name(levels), X, cardinality, unique)


def mac_address_anonymizer(X: ddDataframe, cardinality: dict, locale: str = "en", unique: bool = True, **kwargs) -> str:
    fake = Faker(locale=locale)
    return anonymizer_from_one(fake.mac_address, X, cardinality, unique)


def port_number_anonymizer(X: ddDataframe, cardinality: dict, locale: str = "en", unique: bool = True, **kwargs) -> str:
    fake = Faker(locale=locale)
    return anonymizer_from_one(fake.port_number, X, cardinality, unique)


def uri_anonymizer(X: ddDataframe, cardinality: dict, locale: str = "en", unique: bool = True, **kwargs) -> str:
    fake = Faker(locale=locale)
    return anonymizer_from_one(fake.uri, X, cardinality, unique)


def url_anonymizer(X: ddDataframe, cardinality: dict, locale: str = "en", unique: bool = True, **kwargs) -> str:
    fake = Faker(locale=locale)
    return anonymizer_from_one(fake.url, X, cardinality, unique)


def user_name_anonymizer(X: ddDataframe, cardinality: dict, locale: str = "en", unique: bool = True, **kwargs) -> str:
    fake = Faker(locale=locale)
    return anonymizer_from_one(fake.user_name, X, cardinality, unique)


def job_anonymizer(X: ddDataframe, cardinality: dict, locale: str = "en", unique: bool = True, **kwargs) -> str:
    fake = Faker(locale=locale)
    return anonymizer_from_one(fake.job, X, cardinality, unique)


def uuid4_anonymizer(X: ddDataframe, cardinality: dict, locale: str = "en", unique: bool = True, **kwargs) -> str:
    fake = Faker(locale=locale)
    return anonymizer_from_one(fake.uuid4, X, cardinality, unique)


def first_name_anonymizer(X: ddDataframe, cardinality: dict, locale: str = "en", unique: bool = True, **kwargs) -> str:
    fake = Faker(locale=locale)
    return anonymizer_from_one(fake.first_name, X, cardinality, unique)


def first_name_female_anonymizer(X: ddDataframe, cardinality: dict, locale: str = "en", unique: bool = True, **kwargs) -> str:
    fake = Faker(locale=locale)
    return anonymizer_from_one(fake.first_name_female, X, cardinality, unique)


def first_name_male_anonymizer(X: ddDataframe, cardinality: dict, locale: str = "en", unique: bool = True, **kwargs) -> str:
    fake = Faker(locale=locale)
    return anonymizer_from_one(fake.first_name_male, X, cardinality, unique)


def last_name_anonymizer(X: ddDataframe, cardinality: dict, locale: str = "en", unique: bool = True, **kwargs) -> str:
    fake = Faker(locale=locale)
    return anonymizer_from_one(fake.last_name, X, cardinality, unique)


def name_anonymizer(X: ddDataframe, cardinality: dict, locale: str = "en", unique: bool = True, **kwargs) -> str:
    fake = Faker(locale=locale)
    return anonymizer_from_one(fake.name, X, cardinality, unique)


def name_female_anonymizer(X: ddDataframe, cardinality: dict, locale: str = "en", unique: bool = True, **kwargs) -> str:
    fake = Faker(locale=locale)
    return anonymizer_from_one(fake.name_female, X, cardinality, unique)


def name_male_anonymizer(X: ddDataframe, cardinality: dict, locale: str = "en", unique: bool = True, **kwargs) -> str:
    fake = Faker(locale=locale)
    return anonymizer_from_one(fake.name_male, X, cardinality, unique)


def phone_number_anonymizer(X: ddDataframe, cardinality: dict, locale: str = "en", unique: bool = True, **kwargs) -> str:
    fake = Faker(locale=locale)
    return anonymizer_from_one(fake.phone_number, X, cardinality, unique)


def ssn_anonymizer(X: ddDataframe, cardinality: dict, locale: str = "en", unique: bool = True, **kwargs) -> str:
    fake = Faker(locale=locale)
    return anonymizer_from_one(fake.ssn, X, cardinality, unique)


def city_anonymizer(X: ddDataframe, cardinality: dict, locale: str = "en", unique: bool = True, **kwargs) -> str:
    fake = Faker(locale=locale)
    return anonymizer_from_one(fake.city, X, cardinality, unique)


def country_anonymizer(X: ddDataframe, cardinality: dict, locale: str = "en", unique: bool = True, **kwargs) -> str:
    fake = Faker(locale=locale)
    return anonymizer_from_one(fake.country, X, cardinality, unique)


def country_code_anonymizer(X: ddDataframe, cardinality: dict, locale: str = "en", unique: bool = True, **kwargs) -> str:
    fake = Faker(locale=locale)
    return anonymizer_from_one(fake.country_code, X, cardinality, unique)


def street_address_anonymizer(X: ddDataframe, cardinality: dict, locale: str = "en", unique: bool = True, **kwargs) -> str:
    fake = Faker(locale=locale)
    return anonymizer_from_one(fake.street_address, X, cardinality, unique)


def street_name_anonymizer(X: ddDataframe, cardinality: dict, locale: str = "en", unique: bool = True, **kwargs) -> str:
    fake = Faker(locale=locale)
    return anonymizer_from_one(fake.street_address, X, cardinality, unique)


def full_address_anonymizer(X: ddDataframe, cardinality: dict, locale: str = "en", unique: bool = True, **kwargs) -> str:
    fake = Faker(locale=locale)
    return anonymizer_from_one(fake.address, X, cardinality, unique)


def credit_card_number_anonymizer(
    X: ddDataframe, cardinality: dict,
    card_type: Optional[
        Literal[
            "amex",
            "diners",
            "discover",
            "jcb",
            "jcb15",
            "jcb16",
            "maestro",
            "mastercard",
            "visa",
            "visa13",
            "visa16",
            "visa19",
        ]
    ] = None, locale: str = "en", unique: bool = True, **kwargs
) -> str:
    fake = Faker(locale=locale)
    return anonymizer_from_one(lambda: fake.credit_card_number(card_type), X, cardinality, unique)


def credit_card_provider_anonymizer(X: ddDataframe, cardinality: dict, locale: str = "en", unique: bool = True, **kwargs) -> str:
    fake = Faker(locale=locale)
    return anonymizer_from_one(fake.credit_card_provider, X, cardinality, unique)


def credit_card_expire_anonymizer(X: ddDataframe, cardinality: dict, locale: str = "en", unique: bool = True, **kwargs) -> str:
    fake = Faker(locale=locale)
    return anonymizer_from_one(fake.credit_card_expire, X, cardinality, unique)


def vat_anonymizer(X: ddDataframe, cardinality: dict, locale: str = "en", unique: bool = True, **kwargs) -> str:
    regex = r"\s*?[a-zA-Z0-9]{2,4}\d{3,7}\w{3}\b"
    return anonymizer_from_one(get_one_regex(regex), X, cardinality, unique)


def phone_anonymizer(X: ddDataframe, cardinality: dict, locale: str = "en", unique: bool = True, **kwargs) -> str:
    fake = Faker(locale=locale)
    return anonymizer_from_one(lambda: fake.msisdn()[3:], X, cardinality, unique)


def int_anonymizer(X: ddDataframe, cardinality: dict, locale: str = "en", unique: bool = True, **kwargs) -> str:
    starting_value = kwargs.get("starting_value", 1)

    def gen():
        i = starting_value
        while True:
            yield i
            i += 1
    f = gen()
    return anonymizer_from_one(lambda: next(f), X, cardinality, unique)


def text_anonymizer(X: ddDataframe, card, text_anonymizers: list, anonymizers, **kwargs):
    def get_anonymizer_name(anonymizer) -> str:
        if isinstance(anonymizer.type, str):
            return anonymizer.type.upper()
        return anonymizer.type.name.upper()

    dummy_df = pdDataFrame({"_": ["_"]})
    dummy_card = {"_": 1}

    def replace_text(text, anonymizer_config):
        anonymizer_list = anonymizer_config.params.get(
            "text_anonymizers",
            [anonymizer_config]
        )
        anonymizer_list = sorted(
            anonymizer_list, key=lambda x: x.params["end"], reverse=True)

        for config in anonymizer_list:
            prefix = text[:config.params["start"]]
            suffix = text[config.params["end"]:]
            replaced = anonymizers[get_anonymizer_name(config)].anonymizer(
                dummy_df,  # dummy dataset
                dummy_card,  # dummy cardinality
                # locale and other params
                **anonymizers[get_anonymizer_name(config)].params
            )["_"].loc[0]
            text = f"{prefix}{replaced}{suffix}"
        return text

    anonymized = [
        replace_text(row, anon)
        for row, anon in zip(X[X.columns[0]], text_anonymizers)
    ]
    X[X.columns[0]] = ddarray(anonymized)
    return X
