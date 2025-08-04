from __future__ import absolute_import, division, print_function

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Tuple
from uuid import uuid1

from dask.distributed import Client
from dask_gateway import GatewayCluster

from ydata.utils.dask import DaskCluster
from ydata.dataset.filetype import FileType

from ydata.connectors.exceptions import DataConnectorsException
from ydata.connectors.storages import (_AZURE_STORAGE, _BIGQUERY_STORAGE, _GCS_STORAGE, _LOCAL_STORAGE, _RDBMS_STORAGE,
                                       _S3_STORAGE)

from ydata._licensing import licensed


class BaseConnector(ABC):
    """Base connectors interface."""

    STORAGE_TYPE = None


    @licensed
    def __init__(self):
        self.id = uuid1()
        self.creation_date = datetime.utcnow()
        self.cluster, self.cluster_client = self.__init_dask()
        self.storage_options = None

    def __init_dask(self) -> Tuple[Optional[GatewayCluster], Client]:
        dask_cluster = DaskCluster()
        return (dask_cluster.cluster, dask_cluster.client)

    def set_env_vars(self):
        """Set authentication and access of the current store to the env
        vars."""
        pass

    @property
    def is_local_store(self):
        return self.STORAGE_TYPE == _LOCAL_STORAGE

    @property
    def is_s3_store(self):
        return self.STORAGE_TYPE == _S3_STORAGE

    @property
    def is_azure_store(self):
        return self.STORAGE_TYPE == _AZURE_STORAGE

    @property
    def is_gcs_store(self):
        return self.STORAGE_TYPE == _GCS_STORAGE

    @property
    def is_bigquery_store(self):
        return self.STORAGE_TYPE == _BIGQUERY_STORAGE

    @property
    def is_rdbms_store(self):
        return self.STORAGE_TYPE == _RDBMS_STORAGE

    @abstractmethod
    def test(self):
        pass

    @staticmethod
    def _rm_unnamed(df):
        if "Unnamed: 0" in df.columns:
            df = df.drop("Unnamed: 0", axis=1)
        return df

    @staticmethod
    def parse_file_type(filetype):
        if isinstance(filetype, FileType):
            return filetype
        elif isinstance(filetype, str):
            return FileType(filetype)
        else:
            raise DataConnectorsException(f"Invalid file type {filetype}")

    @staticmethod
    def check_file_extension(filename):
        try:
            extension = filename.split(".")[1]
        except Exception:
            extension = None
        return extension
