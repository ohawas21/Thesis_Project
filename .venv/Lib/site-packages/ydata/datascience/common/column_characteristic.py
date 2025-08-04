from ydata.core.enum import StringEnum


class ColumnCharacteristic(StringEnum):
    # Adding new characteristics may require adding a new generator to:
    # src/ydatasynthesizers/ydata-synthesizers/ydata/synthesizers/faker/utils.py
    ID = "id"
    EMAIL = "email"
    URL = "url"
    UUID = "uuid"
    NAME = "name"
    PHONE = "phone"
    VAT = "vat"
    IBAN = "iban"
    CREDIT_CARD = "credit_card"
    COUNTRY = "country"
    ZIPCODE = "zipcode"
    ADDRESS = "address"
    PII = 'PII'
    # Generic characteristic grouping Country, Address, etc.
    LOCATION = 'location'
    PERSON = 'person'  # Generic characteristic grouping person related attributes
