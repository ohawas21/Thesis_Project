import gc
from os import environ

from dask.distributed import Client
from dask_gateway import GatewayCluster
from dask_gateway.options import Options, String

from ydata.dataset.settings import RUNNING_ENV


class DaskCluster:
    """Singleton class used to store the Dask cluster reference and its
    auxiliary functions."""

    _DEFAULT_MIN_NUMBER_OF_WORKERS = 1

    def __init__(self) -> None:
        self._cluster, self._cluster_client = self.__init_dask()

    def __init_dask(self) -> tuple[GatewayCluster | None, Client]:
        if RUNNING_ENV == "LOCAL":
            client = Client()
            return (client.cluster, client)
        else:
            namespace = environ.get("NAMESPACE")
            options = Options(String("namespace", namespace, label="Project")) if namespace else Options()
            cluster = GatewayCluster(cluster_options=options)
            cluster.adapt(active=True)
            return (cluster, cluster.get_client())


    def memory_collect(self):
        self.client.run(gc.collect)

    @property
    def cluster(self):
        return self._cluster

    @property
    def client(self):
        return self._cluster_client
