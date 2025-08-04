import os

RUN_STORES_ACCESS_KEYS = os.environ.get(
    "STORAGES_RUN_STORES_ACCESS_KEYS", {}
)
TMP_AUTH_GCS_ACCESS_PATH = os.environ.get(
    "STORAGES_TMP_AUTH_GCS_ACCESS_PATH",
    "/tmp/.ydata/.gcsaccess.json",
)
