import os
from json import loads as json_loads
from pathlib import Path

from pandas import read_csv

from ydata.connectors import GCSConnector
from ydata.dataset.filetype import FileType
from ydata.utils.formats import read_json


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent.parent.parent.parent


def get_gcs_config_path() -> Path:
    return get_project_root() / "test-utils" / "gcs_paths.yml"


def get_gcs_data_path() -> Path:
    return get_project_root() / "data" / "gcs"


def get_local_gcs_token() -> Path:
    return get_project_root() / ".secrets" / "gcs_credentials.json"


def gcs_cache_database(database: str, gcs_tables_paths: list[str], gcs_schema_path: str):
    data_path = get_gcs_data_path()
    data_path.mkdir(exist_ok=True)
    database_path = data_path / database
    database_path.mkdir(exist_ok=True)

    token = json_loads(os.environ["gcs_credentials"]) \
        if "gcs_credentials" in os.environ \
        else read_json(get_local_gcs_token())
    connector = GCSConnector("ydatasynthetic", keyfile_dict=token)
    tables = {}
    for gcs_path in gcs_tables_paths:
        file_name = os.path.basename(gcs_path)
        file_path = database_path / file_name
        if not file_path.exists():
            dataset = connector.read_file(gcs_path, file_type=FileType.CSV)
            df = dataset.to_pandas()
            df.to_csv(file_path, index=False)
        else:
            df = read_csv(file_path)
        table_name = file_name.split(".")[0]
        tables[table_name] = df

    file_path = database_path / os.path.basename(gcs_schema_path)
    if not file_path.exists():
        gcs_client = connector.client
        bucket_name, blob = connector.parse_connector_url(gcs_schema_path)
        bucket = gcs_client.bucket(bucket_name)
        blob_obj = bucket.blob(blob)
        blob_obj.download_to_filename(file_path)
    schema_json = read_json(file_path)

    return tables, schema_json
