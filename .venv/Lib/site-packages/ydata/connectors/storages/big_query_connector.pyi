from _typeshed import Incomplete
from ydata.connectors.base_connector import BaseConnector
from ydata.dataset.dataset import Dataset
from ydata.dataset.engines import VALID_ENGINES as VALID_ENGINES

class BigQueryConnector(BaseConnector):
    """Google BIG QUERY storage connector.

    Attributes:
        client (bigquery.client.Client): Client to bundle configuration needed for API requests.
        project_id (str): ID of the Google Cloud Platform project.
        datasets (List[str]): list of available datasets in the project.
    """
    STORAGE_TYPE: Incomplete
    credentials: Incomplete
    def __init__(self, project_id: Incomplete | None = None, gcs_credentials: Incomplete | None = None, key_path: Incomplete | None = None, keyfile_dict: Incomplete | None = None, scopes: Incomplete | None = None) -> None: ...
    @property
    def project_id(self):
        """str: ID of the GCP project."""
    @property
    def client(self):
        """bigquery.client.Client: BigQuery Client to bundle configuration needed for API requests."""
    def set_client(self) -> None: ...
    def set_env_vars(self) -> None: ...
    def dataset_exist(self, dataset: str) -> bool:
        """Boolean to indicate whether a dataset is available."""
    def table_exist(self, table: str, dataset: str) -> bool:
        """Boolean to indicate whether a table belongs to a dataset."""
    @property
    def datasets(self) -> list[str]:
        """List[str]: List with the name of the available datasets."""
    def set_datasets(self) -> None: ...
    def list_tables(self, dataset: str) -> list[str]:
        """List[str]: Gets the tables under a given dataset."""
    def get_or_create_dataset(self, dataset: str):
        """Creates a new dataset using the BigQuery client if it doesn't exist already."""
    def delete_table_if_exists(self, dataset: str, table: str):
        """Deletes a table from BigQuery if it exists."""
    def delete_dataset_if_exists(self, dataset: str):
        """Deletes a dataset from BigQuery if it exists."""
    def table_schema(self, dataset: str, table: str):
        """
        Get the information about the table
        Args:
            dataset_name: 'str' Name of the dataset
            table_name: 'str' Name of the table
        Returns:
            'list'. List of dicts with the table's metadata
        """
    def query(self, query, n_sample: int | None = None):
        '''
        Extract data from a database through a query
        Args:
            query (str): Receives a query as an string.
                         The table name of sql query has to be of the form "dataset_name.table_name"
            n_sample (Optional[int]): if specified,

        Returns:
            Dataset: A distributed dataset with the query results
        '''
    def query_sample(self, query: str, n_sample=...):
        '''Extract sample data from a database through a query.

        Args:
            query (str): Receives a query as an string.
                         The table name of sql query has to be of the form "dataset_name.table_name"
            n_sample (int): Number of sample size.
        Returns: \'ydata.Dataset\'. A distributed dataset with the query results
        '''
    def write_query_to_gcs(self, query: str, path: str, tmp_dataset: str = 'ydata_tmp', tmp_table: str = 'tmp_query_table', clean_tmp: bool = True):
        """Store a BigQuery query to a Google Cloud Storage.

        BigQuery API does not support outputting BigQuery query results directly to GCS.
        To overcome this issue, we implement a two-stage approach, storing first the query
        results in BigQuery temporary storage (tmp_dataset.tmp_table) and then moving such
        table directly to the provided GCS path.

        Args:
            query (str): Query to execute on BigQuery which will be exported into GCS.
            path (str): File path in GCS in which to write the BigQuery query results.
            tmp_dataset (str): Name of temporary dataset to store intermediate results from query.
            tmp_table (str): Name of temporary table to store intermediate results from query.
            clean_tmp (bool): Whether to drop temporary query table and datasets from BigQuery.
                              Deletes dataset if it doesn't contain any other temporary tables.
        """
    def export_table_to_gcs(self, dataset: str, table: str, bucket: str, filename: str, compression=..., destination_format=...):
        """Exports a BigQuery table into Google Cloud Storage.

        Original documentation following https://cloud.google.com/bigquery/docs/exporting-data.

        Args:
            dataset (str): Name of BigQuery dataset where the table is stored
            table (str): Name of the BigQuery table , from which to export
            bucket (str): GCS bucket name
            filename (str): Name of the target file, under the bucket, in GCS.
            compression (Compression, optional): The compression type to use for exported files.
            destination_format (DestinationFormat, optional): The exported file format.
        """
    def write_table_from_data(self, data: Dataset | VALID_ENGINES, database: str, table: str):
        """Writes a BigQuery table from a given data object.

        Args:
            data (Union[Dataset, VALID_ENGINES]): Original data to be persisted.
            dataset: The name of target BigQuery dataset, in which data will be written.
            table: The name of target BigQuery table, in which data will be written.
        """
    def write_table_from_query(self, query: str, dataset: str, table: str, write_disposition: Incomplete | None = None):
        """Writes a BigQuery table from a BigQuery query. The new result will
        be written to project_id.dataset_name.table_name as specified by the
        connector.

        Args:
            query (str): SQL-like query logic
            dataset_name (str): name of target dataset
            table_name (str): name of target table.
            write_disposition (WriteDisposition, optional): specifies the action that occurs if the destination column
                                                            already exists.
        """
    def test(self) -> None: ...
