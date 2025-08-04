"""
  Enum class for Anonymizer configs
"""
from enum import Enum


class AnonymizerType(Enum):
  REGEX = 1
  IP = 2
  IPV4 = 3
  IPV6 = 4
  HOSTNAME = 5
  LICENSE_PLATE = 6
  ABA = 7
  BANK_COUNTRY = 8
  BBAN = 9
  IBAN = 10
  SWIFT = 11
  BARCODE = 12
  COMPANY = 13
  COMPANY_SUFFIX = 14
  COMPANY_EMAIL = 15
  EMAIL = 16
  DOMAIN_NAME = 17
  MAC_ADDRESS = 18
  PORT_NUMBER = 19
  URI = 20
  USER_NAME = 21
  JOB = 22
  FIRST_NAME = 23
  FIRST_NAME_FEMALE = 24
  FIRST_NAME_MALE = 25
  LAST_NAME = 26
  NAME = 27
  NAME_FEMALE = 28
  NAME_MALE = 29
  SSN = 30
  CITY = 31
  COUNTRY = 32
  COUNTRY_CODE = 33
  STREET_ADDRESS = 34
  STREET_NAME = 35
  FULL_ADDRESS = 36
  URL = 37
  CREDIT_CARD_NUMBER = 38
  CREDIT_CARD_PROVIDER = 39
  CREDIT_CARD_EXPIRE = 40
  VAT = 41
  POSTCODE = 42
  PHONE = 43
  INT = 44
  UUID = 45

  @staticmethod
  def get_anonymizer_type(name):
    """Helper function that receives the anonymizer type either as
    AnonymizerType or its int or string representations and returns the
    anonymizer type. e.g. AnonymizerType.NAME can also be represented by the
    string 'name' or the identifier 27.

    Args:
        name (AnonymizerType | str | int): name or id of the anonymizer type.

    Returns:
        AnonymizerType: A
    """

    if isinstance(name, AnonymizerType):
      return name

    if isinstance(name, str):
      if name.upper() in AnonymizerType._member_names_:
        return AnonymizerType[name.upper()]
      else:
        return None

    if isinstance(name, int):
      return AnonymizerType(name)
