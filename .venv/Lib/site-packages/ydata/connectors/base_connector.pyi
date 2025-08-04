import abc
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from dask.distributed import Client as Client
from dask_gateway import GatewayCluster as GatewayCluster

class BaseConnector(ABC, metaclass=abc.ABCMeta):
    """Base connectors interface."""
    STORAGE_TYPE: Incomplete
    id: Incomplete
    creation_date: Incomplete
    storage_options: Incomplete
    def __init__(self) -> None: ...
    def set_env_vars(self) -> None:
        """Set authentication and access of the current store to the env
        vars."""
    @property
    def is_local_store(self): ...
    @property
    def is_s3_store(self): ...
    @property
    def is_azure_store(self): ...
    @property
    def is_gcs_store(self): ...
    @property
    def is_bigquery_store(self): ...
    @property
    def is_rdbms_store(self): ...
    @abstractmethod
    def test(self): ...
    @staticmethod
    def parse_file_type(filetype): ...
    @staticmethod
    def check_file_extension(filename): ...
