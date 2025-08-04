from delta_sharing.protocol import Schema, Share
from delta_sharing.rest_client import DeltaSharingProfile
from pathlib import Path
from typing import BinaryIO, TextIO
from ydata.connectors.base_connector import BaseConnector
from ydata.dataset import Dataset

def manage_exceptions(func): ...

class DatabricksUnityCatalog(BaseConnector):
    """Databrick Unity catalog connector Allows the user to check all the
    exiting share spaces from Delta sharing, schemas and tables."""
    def __init__(self, profile: str | BinaryIO | TextIO | Path | DeltaSharingProfile) -> None: ...
    @property
    def client(self):
        """Creates a client connection to Delta Sharing."""
    def set_client(self) -> None:
        """Sets up a new connection to Delta sharing through a provided
        profile_file."""
    def list_shares(self) -> list:
        """List all the available shares for a profile."""
    def list_schemas(self, share_name: str | Share) -> list:
        """List all the available schemas given a share."""
    def list_tables(self, share_name: str | Share, schema_name: str | Schema) -> list:
        """List all the available tables given a schema."""
    def list_all_tables(self) -> dict:
        """List all the available tables for a given profile_config.

        It returns a list of Tables with share and schema information
        """
    def read_table_sample(self, share_name: str, schema_name: str, table_name: str, sample_size: int = 100) -> Dataset: ...
    def read_table(self, share_name: str, schema_name: str, table_name: str, sample_size: int | None = None) -> Dataset:
        """Load the table as a Dataset.

        :param share_name: name of the share the table belong to
        :param schema_name: name of the schema that the table belong to
        :param table_name: name of the table
        :param limit: number of rows to be load into the Dataset
        :return: a Dataset object with the loaded data
        """
    def test(self) -> None: ...
