from os import getenv

from pandas import CategoricalDtype
from pandas import DataFrame as pdDataFrame
from pandas import Series as pdSeries

from ydata.dataset.multidataset import MultiDataset
from ydata.metadata import Metadata

from ydata.synthesizers.exceptions import SynthesizerValueError
from ydata.synthesizers.logger import synthlogger_config
from ydata.synthesizers.calculated_features import CalculatedFeature

# Define here the logging status and definition
logger = synthlogger_config(verbose=getenv(
    "VERBOSE", "false").lower() == "true")


def init_calculated_features(calculated_features: dict | None):
    """Initialize calculated features.

    Args:
        calculated_features (dict | None): calculated feaures configuration.

    Returns:
        list(CalculatedFeature): list of calculated features.
    """
    if calculated_features is not None:
        return [
            CalculatedFeature.from_dict(cf)
            for cf in calculated_features
        ]
    else:
        return []


def validate_calculated_features(data: MultiDataset, calculated_features: list[CalculatedFeature]) -> None:
    """Validate calculated features definition.

    Raises ValueError if any incossistent calculated features is found.

    Args:
        data (MultiDataset): dataset were the calculated features will be applied.
        calculated_features (list[CalculatedFeature]): list of calculated features.

    Raises:
        ValueError: error indication the invalid calculated feature
    """
    if len(calculated_features) == 0:
        return

    tables = list(data.keys())

    def raise_error(msg: str):
        logger.error(msg)
        raise SynthesizerValueError(msg)

    def validate_feature(cf: CalculatedFeature, feat: str, field: str, check_column: bool = True):
        if "." not in feat:
            raise_error(
                f" Ivalid calculated feature configuration provided. "
                f"{cf} {field} [{feat}] format should be `table_name.column_name`."
                f"Please revisit the provided configuration.")
        table_name, column = feat.split('.', 1)
        if table_name not in tables:
            raise_error(
                f"Invalid table name provided in the calculated features configuration."
                f"{cf} {field} [{feat}] table name [{table_name}] does not exist in the dataset schema.")
        if check_column and column not in data[table_name].columns:
            raise_error(
                f"Invalid column name provided in the calculated features configuration."
                f"{cf} {field} [{feat}] column [{column}] not present in [{table_name}]")

    for cf in calculated_features:
        for feat in cf.features:
            validate_feature(cf, feat, "features", check_column=False)

        for feat in cf.calculated_from:
            validate_feature(
                cf, feat, "calculated_from", check_column=True)

        feature_tables = get_tables_from_columns(cf.features)
        if len(feature_tables) != 1:
            raise_error(
                f"Invalid calculated features configuration provided."
                f"Multiple features calculation is only allowed for features within same table [{cf}]."
                f"Please revisit the provided configuration.")


def drop_calculated_features_columns(
    calculated_features: list[CalculatedFeature],
    tables_df: dict[str, pdDataFrame]
):
    """Remove calculated features columns from the data.

    Obs. modify `tables_df` inplace.

    Args:
        calculated_features (list[CalculatedFeature]): list of calculated features.
        tables_df (dict[str, pdDataFrame]): data were the calculated features will be applied.

    Returns:
        dict[str, pdDataFrame]: tables' data with columns removed
    """
    calc_features_per_table = {}
    for ct in calculated_features:
        table, _ = ct.features[0].split('.', 1)
        if table not in calc_features_per_table:
            calc_features_per_table[table] = []
        calc_features_per_table[table].extend(ct.features)

    for k, df in tables_df.items():
        if k in calc_features_per_table:
            columns_to_drop = [
                c for c in calc_features_per_table[k]
                if c in df.columns
            ]
            df.drop(columns=columns_to_drop)
    return tables_df


def drop_table_calculated_features_columns(
    table: str,
    table_data: pdDataFrame,
    table_metadata: Metadata,
    calculated_features: list[CalculatedFeature],
):
    """Remove calculated features columns from the data.

    Args:
        calculated_features (list[CalculatedFeature]): list of calculated features.

    Returns:
        dict[str, pdDataFrame]: tables' data with columns removed
    """
    calc_features = []
    for ct in calculated_features:
        cf_table, _ = ct.features[0].split('.', 1)
        if cf_table != table:
            continue
        calc_features.extend(ct.features)

    if not calc_features:
        return table_data, table_metadata

    columns_to_drop = [
        c for c in calc_features
        if c in table_data.columns
    ]
    if columns_to_drop:
        table_data = table_data.drop(columns=columns_to_drop)

    return table_data, table_metadata


def get_tables_from_columns(columns: list[str]) -> set[str]:
    tables = {
        f.split(".", 1)[0]
        for f in columns
    }
    return tables


def is_intra_table_calculated_feature(
    calculated_feature: CalculatedFeature,
) -> bool:
    feature_tables = get_tables_from_columns(
        calculated_feature.features)
    calc_from = get_tables_from_columns(
        calculated_feature.calculated_from)
    if len(feature_tables) != 1 or len(calc_from) != 1:
        return False
    feature_table = feature_tables.pop()
    source_table = calc_from.pop()
    if feature_table != source_table:
        return False
    return True


def _convert_series_dtype(series: pdSeries) -> pdSeries:
    """Convert categorical series to the category type.

    Args:
        series (pdSeries): pandas series.

    Returns:
        pdSeries: pandas series with converted type.
    """
    if series.dtype == 'categorical' or isinstance(series.dtype, CategoricalDtype):
        dtype = series.cat.categories.dtype
        dtype = "str" if dtype == "object" else dtype
        return series.astype(dtype)
    return series


def _apply_intra_table_calculated_feature(calculated_feature: CalculatedFeature, table_data: pdDataFrame) -> pdDataFrame:
    source_data = []
    for table_col in calculated_feature.calculated_from:
        _, col_name = table_col.split(".", 1)
        source_data.append(table_data[col_name])

    source_data = [_convert_series_dtype(sd) for sd in source_data]
    calc_data = calculated_feature.function(*source_data)
    for ix, table_col in enumerate(calculated_feature.features):
        _, col_name = table_col.split(".", 1)
        table_data[col_name] = \
            calc_data[:, ix] if calc_data.ndim > 1 else calc_data

    return table_data


def _get_source_data(source: str, sample_tables: dict[str, pdDataFrame]) -> dict[str, pdSeries]:
    table, column = source.split(".", 1)
    assert table in sample_tables
    assert column in sample_tables[table]
    return {column: sample_tables[table][column]}


def _apply_inter_table_calculated_feature(
    calculated_feature: CalculatedFeature,
    table_data: pdDataFrame,
    sample_tables: dict[str, pdDataFrame]
) -> pdDataFrame:
    # validate the data source for the computation comes from a single table
    tables = get_tables_from_columns(calculated_feature.calculated_from)
    assert len(tables) == 1
    source_table = tables.pop()

    # validate the features being calcualted comes from a single table
    tables = get_tables_from_columns(calculated_feature.features)
    assert len(tables) == 1
    feature_table = tables.pop()

    source_data = {}
    for column in calculated_feature.calculated_from:
        source_data.update(_get_source_data(column, sample_tables))

    feature_references = []
    source_references = []
    for (left, right) in calculated_feature.reference_keys:
        left_table, left_column = left.split(".", 1)
        right_table, right_column = right.split(".", 1)

        # validate that the reference_keys come from the source and feature table
        # validate that both reference_keys don't come from the same table
        assert left_table in {source_table, feature_table}
        assert right_table in {source_table, feature_table}
        assert left_table != right_table
        assert right_column in sample_tables[right_table]
        feature_references.append(left_column)
        source_references.append(right_column)

        source_data.update(_get_source_data(right, sample_tables))

    source_data = {
        col: _convert_series_dtype(sd)
        for col, sd in source_data.items()
    }
    calc_data = calculated_feature.function(*source_data.values())
    assert isinstance(calc_data, pdDataFrame)
    columns = [col for col in calc_data.columns if col not in source_references]
    assert len(columns) == len(calculated_feature.features)
    columns_map = {
        col: f.split('.', 1)[-1]
        for f, col in zip(calculated_feature.features, columns)
    }
    calc_data = calc_data.rename(columns=columns_map)

    table_data = table_data.merge(
        calc_data,
        how='left',
        left_on=feature_references,
        right_on=source_references,
        suffixes=("_todrop_", None)
    )
    cols = [col for col in table_data.columns if not col.endswith("_todrop_")]
    return table_data[cols]


def apply_table_calculated_features(
    calculated_features: list[CalculatedFeature],
    table: str,
    table_data: pdDataFrame,
    sample_tables: dict[str, pdDataFrame]
) -> pdDataFrame:
    for calculated_feature in calculated_features:
        feature_tables = get_tables_from_columns(calculated_feature.features)
        if table not in feature_tables:
            continue

        if is_intra_table_calculated_feature(calculated_feature):
            table_data = _apply_intra_table_calculated_feature(
                calculated_feature=calculated_feature,
                table_data=table_data,
            )
        else:
            table_data = _apply_inter_table_calculated_feature(
                calculated_feature=calculated_feature,
                table_data=table_data,
                sample_tables=sample_tables,
            )

    return table_data
