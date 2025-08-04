"""Python class to connect to the Unity Catalog."""
import json
from pathlib import Path
from typing import BinaryIO, TextIO, Union

from delta_sharing import SharingClient
from delta_sharing.protocol import Schema, Share, Table
from delta_sharing.reader import DeltaSharingReader
from delta_sharing.rest_client import DataSharingRestClient, DeltaSharingProfile
from requests.exceptions import HTTPError

from ydata.connectors.base_connector import BaseConnector
from ydata.connectors.exceptions import CatalogConnectorException, InvalidCatalogTokenException
from ydata.dataset import Dataset


def manage_exceptions(func):
    def func_exceptions(self, *arg, **kwargs):
        try:
            return func(self, *arg, **kwargs)
        except HTTPError as exc:
            status_code = exc.response.status_code
            if status_code == 401:
                raise InvalidCatalogTokenException(
                    "Provided config file is not valid. Please validate your input")
            elif status_code == 404:
                error_message = json.loads(exc.response.text)['message']
                raise CatalogConnectorException(
                    f"Something went wrong: {error_message}. Please validate your input")
            elif status_code == 400:
                error_message = json.loads(exc.response.text)['message']
                raise CatalogConnectorException(
                    f"Something went wrong\n {error_message}.")
            else:
                raise CatalogConnectorException(
                    "There was an error while trying to list your Catalog shares") from exc
        except Exception as exc:
            error_message = json.loads(exc.response.text)['message']
            raise CatalogConnectorException(
                f"There was an error while trying to access your Catalog resources \n {error_message}")

    return func_exceptions


class DatabricksUnityCatalog(BaseConnector):
    """Databrick Unity catalog connector Allows the user to check all the
    exiting share spaces from Delta sharing, schemas and tables."""

    def __init__(self, profile: Union[str, BinaryIO, TextIO, Path, DeltaSharingProfile]):
        super().__init__()
        if not isinstance(profile, DeltaSharingProfile):
            profile = DeltaSharingProfile.read_from_file(profile)
        self._profile = profile
        self._rest_client = DataSharingRestClient(profile)
        self._client = None

    @property
    def client(self):
        """Creates a client connection to Delta Sharing."""
        if self._client is None:
            self.set_client()
        return self._client

    def set_client(self):
        """Sets up a new connection to Delta sharing through a provided
        profile_file."""
        self._client = SharingClient(self._profile)

    @manage_exceptions
    def list_shares(self) -> list:
        """List all the available shares for a profile."""
        return [share.name for share in self.client.list_shares()]

    @manage_exceptions
    def list_schemas(self, share_name: Union[str, Share]) -> list:
        """List all the available schemas given a share."""
        share = Share(share_name)
        return [schema.name for schema in self.client.list_schemas(share=share)]

    @manage_exceptions
    def list_tables(self, share_name: Union[str, Share], schema_name: Union[str, Schema]) -> list:
        """List all the available tables given a schema."""
        if not isinstance(schema_name, Schema):
            schema = Schema(share=share_name, name=schema_name)
        return [table.name for table in self.client.list_tables(schema=schema)]

    @manage_exceptions
    def list_all_tables(self) -> dict:
        """List all the available tables for a given profile_config.

        It returns a list of Tables with share and schema information
        """
        return {table.name: {'share': table.share, 'schema': table.schema} for table in self.client.list_all_tables()}

    def read_table_sample(self, share_name: str, schema_name: str, table_name: str, sample_size: int = 100) -> Dataset:
        return self.read_table(share_name=share_name,
                               schema_name=schema_name,
                               table_name=table_name,
                               sample_size=sample_size
                               )

    @manage_exceptions
    def read_table(self, share_name: str, schema_name: str, table_name: str, sample_size: int | None = None) -> Dataset:
        """Load the table as a Dataset.

        :param share_name: name of the share the table belong to
        :param schema_name: name of the schema that the table belong to
        :param table_name: name of the table
        :param limit: number of rows to be load into the Dataset
        :return: a Dataset object with the loaded data
        """

        df = DeltaSharingReader(
            table=Table(name=table_name, share=share_name, schema=schema_name),  rest_client=self._rest_client,
            limit=sample_size
        ).to_pandas()

        return Dataset(df)

    def test(self):
        _ = self.list_shares()
