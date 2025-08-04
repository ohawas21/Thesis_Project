from enum import Enum
from typing import Callable

from ydata.datascience.common import AnonymizerType

from ydata.preprocessors.methods.anonymization import anonymizer as anonymizer_methods

def _get_anonymizer_method(anonymizer_type: AnonymizerType) -> Callable:
    mapping = {
        AnonymizerType.REGEX: anonymizer_methods.regex_anonymizer,
        AnonymizerType.IP: anonymizer_methods.ip_anonymizer,
        AnonymizerType.IPV4: anonymizer_methods.ipv4_anonymizer,
        AnonymizerType.IPV6: anonymizer_methods.ipv6_anonymizer,
        AnonymizerType.HOSTNAME: anonymizer_methods.hostname_anonymizer,
        AnonymizerType.LICENSE_PLATE: anonymizer_methods.license_plate_anonymizer,
        AnonymizerType.ABA: anonymizer_methods.aba_anonymizer,
        AnonymizerType.BANK_COUNTRY: anonymizer_methods.bank_country_anonymizer,
        AnonymizerType.BBAN: anonymizer_methods.bban_anonymizer,
        AnonymizerType.IBAN: anonymizer_methods.iban_anonymizer,
        AnonymizerType.SWIFT: anonymizer_methods.swift_anonymizer,
        AnonymizerType.BARCODE: anonymizer_methods.barcode_anonymizer,
        AnonymizerType.COMPANY: anonymizer_methods.company_anonymizer,
        AnonymizerType.COMPANY_SUFFIX: anonymizer_methods.company_suffix_anonymizer,
        AnonymizerType.COMPANY_EMAIL: anonymizer_methods.company_email_anonymizer,
        AnonymizerType.EMAIL: anonymizer_methods.email_anonymizer,
        AnonymizerType.DOMAIN_NAME: anonymizer_methods.domain_name_anonymizer,
        AnonymizerType.MAC_ADDRESS: anonymizer_methods.mac_address_anonymizer,
        AnonymizerType.PORT_NUMBER: anonymizer_methods.port_number_anonymizer,
        AnonymizerType.URI: anonymizer_methods.uri_anonymizer,
        AnonymizerType.USER_NAME: anonymizer_methods.user_name_anonymizer,
        AnonymizerType.JOB: anonymizer_methods.job_anonymizer,
        AnonymizerType.FIRST_NAME: anonymizer_methods.first_name_anonymizer,
        AnonymizerType.FIRST_NAME_FEMALE: anonymizer_methods.first_name_female_anonymizer,
        AnonymizerType.FIRST_NAME_MALE: anonymizer_methods.first_name_male_anonymizer,
        AnonymizerType.LAST_NAME: anonymizer_methods.last_name_anonymizer,
        AnonymizerType.NAME: anonymizer_methods.name_anonymizer,
        AnonymizerType.NAME_FEMALE: anonymizer_methods.name_female_anonymizer,
        AnonymizerType.NAME_MALE: anonymizer_methods.name_male_anonymizer,
        AnonymizerType.SSN: anonymizer_methods.ssn_anonymizer,
        AnonymizerType.CITY: anonymizer_methods.city_anonymizer,
        AnonymizerType.COUNTRY: anonymizer_methods.country_anonymizer,
        AnonymizerType.COUNTRY_CODE: anonymizer_methods.country_code_anonymizer,
        AnonymizerType.STREET_ADDRESS: anonymizer_methods.street_address_anonymizer,
        AnonymizerType.STREET_NAME: anonymizer_methods.street_name_anonymizer,
        AnonymizerType.FULL_ADDRESS: anonymizer_methods.full_address_anonymizer,
        AnonymizerType.URL: anonymizer_methods.url_anonymizer,
        AnonymizerType.CREDIT_CARD_NUMBER: anonymizer_methods.credit_card_number_anonymizer,
        AnonymizerType.CREDIT_CARD_PROVIDER: anonymizer_methods.credit_card_provider_anonymizer,
        AnonymizerType.CREDIT_CARD_EXPIRE: anonymizer_methods.credit_card_expire_anonymizer,
        AnonymizerType.VAT: anonymizer_methods.vat_anonymizer,
        AnonymizerType.POSTCODE: anonymizer_methods.postcode_anonymizer,
        AnonymizerType.PHONE: anonymizer_methods.phone_anonymizer,
        AnonymizerType.INT: anonymizer_methods.int_anonymizer,
        AnonymizerType.UUID: anonymizer_methods.uuid4_anonymizer,
        "TEXT": anonymizer_methods.text_anonymizer,
    }
    return mapping[anonymizer_type]
