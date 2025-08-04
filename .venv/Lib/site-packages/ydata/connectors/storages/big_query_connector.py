"""Connector to read data from Google's BigQuery."""
from typing import List, Optional, Union

from dask.dataframe import from_pandas
from google.api_core.exceptions import NotFound
from google.auth.exceptions import GoogleAuthError
from google.cloud.bigquery import (Compression, DatasetReference, DestinationFormat, ExtractJobConfig, QueryJobConfig,
                                   QueryPriority, WriteDisposition)
from pandas_gbq.gbq import GenericGBQException, TableCreationError, to_gbq

from ydata.connectors.base_connector import BaseConnector
from ydata.connectors.clients import gc_client
from ydata.connectors.exceptions import DataConnectorsException, NoDataAvailable
from ydata.connectors.storages import _BIGQUERY_STORAGE, _MAX_SAMPLE
from ydata.connectors.storages.gcs_connector import GCSConnector
from ydata.dataset.dataset import Dataset
from ydata.dataset.engines import VALID_ENGINES, to_pandas


class BigQueryConnector(BaseConnector):
    """Google BIG QUERY storage connector.

    Attributes:
        client (bigquery.client.Client): Client to bundle configuration needed for API requests.
        project_id (str): ID of the Google Cloud Platform project.
        datasets (List[str]): list of available datasets in the project.
    """

    STORAGE_TYPE = _BIGQUERY_STORAGE

    def __init__(
        self,
        project_id=None,
        gcs_credentials=None,
        key_path=None,
        keyfile_dict=None,
        scopes=None,
    ):
        BaseConnector.__init__(self)
        self._client = None
        self._datasets = None
        self.credentials = {
            "project_id": project_id or keyfile_dict.get("project_id"),
            "credentials": gcs_credentials,
            "key_path": key_path,
            "keyfile_dict": keyfile_dict,
            "scopes": scopes,
        }
        self.set_client()

    @property
    def project_id(self):
        "str: ID of the GCP project."
        return self.credentials["project_id"]

    @property
    def client(self):
        "bigquery.client.Client: BigQuery Client to bundle configuration needed for API requests."
        if self._client is None:
            self.set_client()
        return self._client

    def set_client(self):
        self._client = gc_client.get_bq_client(**self.credentials)

    def set_env_vars(self):
        pass

    def dataset_exist(self, dataset: str) -> bool:
        "Boolean to indicate whether a dataset is available."
        self.set_datasets()  # ensure list is updated
        return dataset in self.datasets

    def table_exist(self, table: str, dataset: str) -> bool:
        "Boolean to indicate whether a table belongs to a dataset."
        return table in self.list_tables(dataset)

    @property
    def datasets(self) -> List[str]:
        "List[str]: List with the name of the available datasets." ""
        if self._datasets is None:
            self.set_datasets()
        return self._datasets

    def set_datasets(self):
        self._datasets = [ds.dataset_id for ds in self.client.list_datasets()]

    def list_tables(self, dataset: str) -> List[str]:
        "List[str]: Gets the tables under a given dataset."
        if self.dataset_exist(dataset):
            return [t.table_id for t in self.client.list_tables(dataset)]
        else:
            raise DataConnectorsException(
                f"It's not possible to access {self.project_id} project available tables for the {dataset}."
            )

    def get_or_create_dataset(self, dataset: str):
        "Creates a new dataset using the BigQuery client if it doesn't exist already."
        if not self.dataset_exist(dataset):
            self.client.create_dataset(dataset)

    def delete_table_if_exists(self, dataset: str, table: str):
        "Deletes a table from BigQuery if it exists."
        if self.table_exist(table=table, dataset=dataset):
            table_ref = DatasetReference(self.project_id, dataset).table(table)
            self.client.delete_table(table=table_ref)

    def delete_dataset_if_exists(self, dataset: str):
        "Deletes a dataset from BigQuery if it exists."
        if self.dataset_exist(dataset=dataset):
            dataset_ref = DatasetReference(self.project_id, dataset)
            self.client.delete_dataset(dataset_ref)

    def table_schema(self, dataset: str, table: str):
        """
        Get the information about the table
        Args:
            dataset_name: 'str' Name of the dataset
            table_name: 'str' Name of the table
        Returns:
            'list'. List of dicts with the table's metadata
        """
        dataset_exists = self.dataset_exist(dataset)
        table_exists = self.table_exist(table, dataset)

        if dataset_exists and table_exists:
            table_id = f"{self.project_id}.{dataset}.{table}"
            table = self.client.get_table(table_id)

            # table.schema contains a list of SchemaField
            # .to_api_repr() returns a dict representation of SchemaField
            table_schema = [f.to_api_repr() for f in table.schema]
            return table_schema
        else:
            raise DataConnectorsException(
                f"{table} does not exist in {dataset} from {self.project_id}. Please validate the provided details."
            )

    def query(self, query, n_sample: Optional[int] = None):
        """
        Extract data from a database through a query
        Args:
            query (str): Receives a query as an string.
                         The table name of sql query has to be of the form "dataset_name.table_name"
            n_sample (Optional[int]): if specified,

        Returns:
            Dataset: A distributed dataset with the query results
        """
        # TODO: Replace hardcoded n_partitions estimation for sample
        # Logic to estimate n_partitions cannot be isolated into a helper method because client object is not pickable.

        job_config = QueryJobConfig(
            priority=QueryPriority.BATCH, allow_large_results=True
        )
        try:
            if n_sample:  # if sample
                query_job = self.client.query(
                    query=query, job_config=job_config)
                result = query_job.result(
                    max_results=n_sample)  # define max results
                df = from_pandas(result.to_dataframe(),
                                 npartitions=20)  # load data
            else:  # if full
                job_config.dry_run = True  # estimate costs
                job_config.use_query_cache = False  # avoid cache
                query_job = self.client.query(
                    query=query, job_config=job_config
                )  # dry_run query
                n_partitions = max(
                    1, int(query_job.total_bytes_processed / 1e8)
                )  # get estimation of partitions
                job_config.dry_run = False  # prepare actual query
                query_job = self.client.query(
                    query=query, job_config=job_config
                )  # real query
                df = from_pandas(
                    query_job.to_dataframe(progress_bar_type="tqdm"),
                    npartitions=n_partitions,
                )
        except NotFound:
            raise NoDataAvailable(
                "The query requires data assets that are not available. \
Please check the project id and project contents."
            )
        except GoogleAuthError:
            raise DataConnectorsException(
                "The provided credentials are not valid.")
        except Exception as exc:
            raise DataConnectorsException from exc
        return Dataset(df=df)

    def query_sample(self, query: str, n_sample=_MAX_SAMPLE):
        """Extract sample data from a database through a query.

        Args:
            query (str): Receives a query as an string.
                         The table name of sql query has to be of the form "dataset_name.table_name"
            n_sample (int): Number of sample size.
        Returns: 'ydata.Dataset'. A distributed dataset with the query results
        """
        return self.query(query=query, n_sample=n_sample)

    def write_query_to_gcs(
        self,
        query: str,
        path: str,
        tmp_dataset: str = "ydata_tmp",
        tmp_table: str = "tmp_query_table",
        clean_tmp: bool = True,
    ):
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
        bucket, filename = GCSConnector(
            **self.credentials).parse_connector_url(path)

        # Create dataset if it doesn't exist
        self.get_or_create_dataset(tmp_dataset)

        # Write query results to temporary dataset.table target
        # Since table is temporary, WRITE_TRUNCATE overwrites if table already exists.
        self.write_table_from_query(
            query=query,
            dataset=tmp_dataset,
            table=tmp_table,
            write_disposition=WriteDisposition.WRITE_TRUNCATE,
        )

        # Move from BigQuery table to GCS
        self.export_table_to_gcs(
            dataset=tmp_dataset, table=tmp_table, bucket=bucket, filename=filename
        )

        # Delete temporary results.
        if clean_tmp:
            self.delete_table_if_exists(dataset=tmp_dataset, table=tmp_table)
            if not self.list_tables(dataset=tmp_dataset):  # no tables for dataset
                self.delete_dataset_if_exists(dataset=tmp_dataset)

    def export_table_to_gcs(
        self,
        dataset: str,
        table: str,
        bucket: str,
        filename: str,
        compression=Compression.GZIP,
        destination_format=DestinationFormat.CSV,
    ):
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
        destination_uri = f"gs://{bucket}/{filename}"
        table_ref = DatasetReference(self.project_id, dataset).table(table)

        # Define the job configuration
        extract_job_config = ExtractJobConfig(
            compression=compression,  # None, 'GZIP', 'DEFLATE', 'SNAPPY',
            destination_format=destination_format,  # CSV, NEWLINE_DELIMITED_JSON, AVRO
        )

        try:
            # Define the job
            extract_job = self.client.extract_table(
                source=table_ref,
                destination_uris=destination_uri,
                job_config=extract_job_config,
            )
            extract_job.result()  # wait for completion
            if extract_job.done():
                print(
                    f'Successfully exported BigQuery Table "{dataset}.{table}" to GCS "{destination_uri}".'
                )
        except Exception as exc:
            raise DataConnectorsException(
                f'Failed to write "{dataset}.{table}" BigQuery table to "{destination_uri}" GCS path.'
            ) from exc

    def write_table_from_data(
        self, data: Union[Dataset, VALID_ENGINES], database: str, table: str
    ):
        """Writes a BigQuery table from a given data object.

        Args:
            data (Union[Dataset, VALID_ENGINES]): Original data to be persisted.
            dataset: The name of target BigQuery dataset, in which data will be written.
            table: The name of target BigQuery table, in which data will be written.
        """
        table_id = f"{database}.{table}"

        # Enforce a Pandas Dataframe to use Pandas GBQ package
        df = data.to_pandas() if isinstance(data, Dataset) else to_pandas(data)

        try:
            to_gbq(
                df,
                table_id,
                project_id=self.project_id,
                credentials=self.client._credentials,
                if_exists="append",
                progress_bar=True,
            )
        except GenericGBQException as exc:
            raise DataConnectorsException(
                "Access denied. "
                f"User does not have valid access to write to table in {dataset} from {self.project_id}. "
                "Please validate provided credentials."
            ) from exc
        except TableCreationError:
            raise DataConnectorsException(
                f"Could not create the {table} table because it already exists."
            )

    def write_table_from_query(
        self, query: str, dataset: str, table: str, write_disposition=None
    ):
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
        job_config = QueryJobConfig(
            destination=f"{self.project_id}.{dataset}.{table}",
            write_disposition=write_disposition or WriteDisposition.WRITE_EMPTY,
        )
        self.client.query(query, job_config=job_config).result()

    def test(self):
        _ = self.client.list_datasets()
