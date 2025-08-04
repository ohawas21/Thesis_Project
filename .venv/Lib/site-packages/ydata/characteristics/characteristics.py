from ydata.characteristics import ColumnCharacteristic
from ydata.metadata import Metadata
from ydata.metadata.multimetadata import MultiMetadata
from ydata.preprocessors.methods.anonymization import AnonymizerType
from ydata.utils.data_types import DataType

CHARAC_TO_ANONYM = {
    ColumnCharacteristic.ID: AnonymizerType.INT,
    ColumnCharacteristic.URL: AnonymizerType.URL,
    ColumnCharacteristic.EMAIL: AnonymizerType.EMAIL,
    ColumnCharacteristic.NAME: AnonymizerType.NAME,
    ColumnCharacteristic.COUNTRY: AnonymizerType.COUNTRY,
    ColumnCharacteristic.ZIPCODE: AnonymizerType.POSTCODE,
    ColumnCharacteristic.ADDRESS: AnonymizerType.FULL_ADDRESS,
    ColumnCharacteristic.IBAN: AnonymizerType.IBAN,
    ColumnCharacteristic.VAT: AnonymizerType.VAT,
    ColumnCharacteristic.CREDIT_CARD: AnonymizerType.CREDIT_CARD_NUMBER,
    ColumnCharacteristic.PHONE: AnonymizerType.PHONE
}
CHARAC_GROUPS = {
    ColumnCharacteristic.PERSON: [ColumnCharacteristic.NAME],
    ColumnCharacteristic.LOCATION: [ColumnCharacteristic.ADDRESS,
                                    ColumnCharacteristic.COUNTRY, ColumnCharacteristic.ZIPCODE]
}


def _suggest_anonymizer_config_metadata(metadata: Metadata) -> dict[str, list[str]]:
    """Suggest anonymizers based on the characteristics.

    Args:
        metadata (Metadata): dataset's metadata

    Returns:
        Dictionary mapping columns to a a list of anonymizers
    """
    config = {}

    candidate_cols = {
        c.name: list(c.characteristics) for c in metadata.columns.values() if c.characteristics
    }

    for c, info in candidate_cols.items():
        anonymizer_groups = [CHARAC_GROUPS[e]
                             for e in info if e in CHARAC_GROUPS]
        anonymizer_candidates = [CHARAC_TO_ANONYM[e]
                                 for e in info if e in CHARAC_TO_ANONYM]
        if anonymizer_candidates:
            # We assume that the importance order is the reverse order because if a characteristic is manually added, it will be at the end
            config[c] = anonymizer_candidates[::-1]

        for group in anonymizer_groups:
            # If a more precise tag is already present, skip this group
            if c in config and any(CHARAC_TO_ANONYM[e] in config[c] for e in group):
                break
            for e in group:
                if c not in config or CHARAC_TO_ANONYM[e] not in config[c]:
                    if c not in config:
                        config[c] = []
                    config[c].append(CHARAC_TO_ANONYM[e])

        # No anonymization for the PII tag based on other tags, so let's suggest a basic INT anonymization
        if c not in config and ColumnCharacteristic.PII in info:
            config[c] = [AnonymizerType.INT]

    full_config = {}
    for col, anom_types in config.items():
        full_config[col] = [{"type": t} for t in anom_types]

    return full_config


def _suggest_anonymizer_config_multimetadata(metadata: MultiMetadata) -> dict[str, dict[str, list[str]]]:
    """Suggest anonymizers based on the characteristics for a multimedata
    object.

    Args:
        MultiMetadata (Metadata): dataset's metadata

    Returns:
        Dictionary mapping tables to anonymizer suggestions.
    """
    config = {}
    for t, m in metadata.items():
        config_table = {}
        if m is not None:
            config_table = suggest_anonymizer_config(m)
        # For each PK add the int anonymizer (will be removed in the MT)
        for pk in metadata.schema[t].primary_keys:
            if pk not in config_table:
                config_table[pk] = []
            config_table[pk].append({'type': AnonymizerType.INT})
        if len(config_table) > 0:
            config[t] = config_table
    return config


def suggest_anonymizer_config(metadata: Metadata | MultiMetadata) -> dict[str, list[str] | dict[str, list[str]]]:
    """Suggest anonymizers based on the characteristics.

    Args:
        metadata (Metadata | MultiMetadata): dataset's metadata

    Returns:
        Dictionary mapping columns to a list of anonymizers
    """
    if isinstance(metadata, MultiMetadata):
        return _suggest_anonymizer_config_multimetadata(metadata)
    return _suggest_anonymizer_config_metadata(metadata)


def deduce_anonymizer_config_for_STR(metadata: Metadata) -> dict[str, str]:
    """Deduce how to generate DataType.STR using the anonymizer.

    Args:
        metadata (Metadata): dataset's metadata

    Returns:
        Dictionary mapping columns to an anonymizer
    """
    config = {}

    # Columns to be anonymized automatically from their characteristics
    candidate_cols = {
        c.name: list(c.characteristics) for c in metadata.columns.values()
        if c.datatype == DataType.STR and c.characteristics
    }

    for c, info in candidate_cols.items():
        anonymizer_candidates = [CHARAC_TO_ANONYM[e]
                                 for e in info if e in CHARAC_TO_ANONYM]
        if anonymizer_candidates:
            # For now we assume that the last characteristics with an anonymizer should be used.
            config[c] = anonymizer_candidates[-1]

    return config


def deduce_anonymizer_config_for_PII(metadata: Metadata) -> dict[str, str]:
    """Deduce how to generate DataType.STR using the anonymizer.

    Args:
        metadata (Metadata): dataset's metadata

    Returns:
        Dictionary mapping columns to an anonymizer
    """
    config = {}

    # Columns to be anonymized automatically from their characteristics
    candidate_cols = {
        c.name: list(c.characteristics) for c in metadata.columns.values() if c.datatype in [DataType.STR, DataType.CATEGORICAL] and ColumnCharacteristic.PII in c.characteristics
    }

    for c, info in candidate_cols.items():
        anonymizer_candidates = [
            CHARAC_TO_ANONYM[e]
            for e in info
            if e in CHARAC_TO_ANONYM
            # STR-ID will not be deduced automatically, it must be added manually
            and e != ColumnCharacteristic.ID
        ]
        if anonymizer_candidates:
            config[c] = anonymizer_candidates[-1]
        else:
            config[c] = AnonymizerType.INT

    return config
