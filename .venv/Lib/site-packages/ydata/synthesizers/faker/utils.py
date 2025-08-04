from os import getenv

from exrex import getone
from faker import Faker
from numpy import nan

from ydata.characteristics import ColumnCharacteristic
from ydata.synthesizers.logger import synthlogger_config

logger = synthlogger_config(verbose=getenv(
    "VERBOSE", "false").lower() == "true")
_MAX_ITER = 100_000


def get_n_unique_from_one(gen, n, max_iter: int | None = None, unique: bool = True):
    if max_iter is None:
        max_iter = max(3 * n, _MAX_ITER)
    res = set()
    it = 0
    while len(res) != n and it < max_iter:
        val = gen()
        res.add(val)
        it += 1
    if it >= max_iter:
        if unique:
            logger.warning(
                f"Could not generate {n} unique values with regex")
        res = list(res)
        missing = n - len(res)
        for _ in range(missing):
            res.append(gen())
    return list(res)


def regex_generator(regex: str):
    return lambda: getone(regex)


def id_generator(initial_value: int = 1):
    def generator():
        i = initial_value
        while True:
            yield i
            i += 1
    id_gen = generator()
    return lambda: next(id_gen)


def vat_generator(locale: str):
    fake = Faker(locale=locale)
    is_valid_locale = True
    # vat_id is only exists for some locales
    try:
        fake.vat_id()
    except Exception:
        is_valid_locale = False

    if is_valid_locale:
        return fake.vat_id
    else:
        # if vat locale is invalid, generate vat-like data using regex
        regex = r"[A-Z]{2}[0-9]{9}"
        return lambda: getone(regex)


def phone_generator(locale: str):
    fake = Faker(locale=locale)
    is_valid_locale = True
    # phone number is only exists for some locales
    try:
        fake.phone_number()
    except Exception:
        is_valid_locale = False

    if is_valid_locale:
        return fake.phone_number
    else:
        fake = Faker(locale="en")
        return fake.phone_number


def get_generator(characteristic: ColumnCharacteristic, locale: str = "en"):
    faker = Faker(locale=locale)
    match characteristic:
        case ColumnCharacteristic.ID:
            return id_generator()
        case ColumnCharacteristic.EMAIL:
            return faker.email
        case ColumnCharacteristic.URL:
            return faker.url
        case ColumnCharacteristic.UUID:
            return faker.uuid4
        case ColumnCharacteristic.NAME:
            return faker.name
        case ColumnCharacteristic.PHONE:
            return phone_generator(locale)
        case ColumnCharacteristic.VAT:
            return faker.vat_id
        case ColumnCharacteristic.IBAN:
            return faker.iban
        case ColumnCharacteristic.CREDIT_CARD:
            return faker.credit_card_number
        case ColumnCharacteristic.COUNTRY:
            return faker.country
        case ColumnCharacteristic.ZIPCODE:
            return faker.postcode
        case ColumnCharacteristic.ADDRESS:
            return faker.street_address
        case ColumnCharacteristic.PERSON:
            return faker.name
    return None


def _convert_characteristic_to_numerical(data: list[str], vartype):
    converted = []
    for val in data:
        if isinstance(val, str):
            if val.isdecimal():
                decimal_str = val
            else:
                decimal_str = "".join([c for c in val if c.isdecimal()])
            value = vartype(decimal_str) if decimal_str else nan
        else:
            value = val
        converted.append(value)
    return converted


def convert_characteristic_to_int(data: list[str]) -> list[int]:
    return _convert_characteristic_to_numerical(data, int)


def convert_characteristic_to_float(data: list[str]) -> list[float]:
    return _convert_characteristic_to_numerical(data, float)
