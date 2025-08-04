from yaml import safe_load

from ydata.connectors.storages.rdbms_connector import Schema
from ydata.dataset.multidataset import MultiDataset

from .cache import gcs_cache_database, get_gcs_config_path


def _get_berka_gcs_paths():
    with open(get_gcs_config_path(), "r") as yml_file:
        gcs_paths = safe_load(yml_file)
        return {
            "schema": gcs_paths["berka_schema_path"],
            "tables": gcs_paths["berka_tables_paths"]
        }


def _get_adventure_works_gcs_paths():
    with open(get_gcs_config_path(), "r") as yml_file:
        gcs_paths = safe_load(yml_file)
        return {
            "schema": gcs_paths["adventure_works_schema_path"],
            "tables": gcs_paths["adventure_works_tables_paths"]
        }


def get_berka() -> MultiDataset:
    berka_paths = _get_berka_gcs_paths()
    tables, schema_json = gcs_cache_database(
        "berka", berka_paths["tables"], berka_paths["schema"])
    schema = Schema(**schema_json)
    schema.tables = {k: v for k, v in schema.tables.items()
                     if k in tables.keys()}
    return MultiDataset(datasets=tables, schema=schema)


def get_adventure_works_6tables() -> MultiDataset:
    adventure_works_paths = _get_adventure_works_gcs_paths()
    tables, schema_json = gcs_cache_database(
        "adventure_works", adventure_works_paths["tables"], adventure_works_paths["schema"])
    schema = Schema(**schema_json)
    schema.tables = {k: v for k, v in schema.tables.items()
                     if k in tables.keys()}
    return MultiDataset(datasets=tables, schema=schema)
