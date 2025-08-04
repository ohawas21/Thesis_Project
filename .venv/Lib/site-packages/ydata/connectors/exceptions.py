from __future__ import absolute_import, division, print_function


class DataConnectorsException(Exception):
    pass


class NoDataAvailable(Exception):
    pass


class InvalidQuery(Exception):
    pass


class InvalidTable(Exception):
    pass


class InvalidIndexCol(Exception):
    pass


class InvalidDatabaseConnection(Exception):
    pass


class InvalidTableException(Exception):
    pass


class GCSPathError(Exception):
    pass


class S3PathError(Exception):
    pass


class InvalidCatalogTokenException(Exception):
    pass


class CatalogConnectorException(Exception):
    pass


class InvalidLakehouseTokenException(Exception):
    pass


class NotFoundLakehouseException(Exception):
    pass


class LakehouseConnectorException(Exception):
    pass
