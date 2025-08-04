"""
    Logger and package metrics
"""
from __future__ import absolute_import, division, print_function

import logging
import sys
import os
import platform
import subprocess

import contextlib

import pandas as pd
import requests

DATATYPE_MAPPING = {
    "RegularSynthesizer": "tabular",
    "TimeSeriesSynthesizer": "timeseries",
    "MultiTableSynthesizer": "multitable",
    "FakerSynthesizer": "fake",
    "DocumentQAGeneration": "text",
    "DocumentGenerator": "text",
}

def utilslogger_config(verbose):
    logger = logging.getLogger("Utils")

    log_file = os.getenv("LOG_FILE")

    if log_file:
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            format="%(levelname)s: %(asctime)s %(message)s",
            level=log_level,
            filename=os.getenv("UTILS_LOG_PATH"),
        )
    else:
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            format="%(levelname)s: %(asctime)s %(message)s",
            level=log_level,
            stream=sys.stdout,
        )

    return logger

def is_running_in_databricks():
    mask = "DATABRICKS_RUNTIME_VERSION" in os.environ
    if "DATABRICKS_RUNTIME_VERSION" in os.environ:
        return os.environ["DATABRICKS_RUNTIME_VERSION"]
    else:
        return str(mask)


def analytics_features(datatype: str,
                       dbx: str,
                       method: str,
                       nrows: int | None = None,
                       ncols: int | None = None,
                       ntables: int | None = None,
                       ndocs: int | None = None) -> None:
    """
        Returns metrics and analytics from ydata-sdk
    """
    endpoint = "https://packages.ydata.ai/ydata-sdk?"

    if bool(os.getenv("YDATA_SDK_NO_ANALYTICS")) is not True:
        try:
            subprocess.check_output("nvidia-smi")
            gpu_present = True
        except Exception:
            gpu_present = False

        python_version = ".".join(platform.python_version().split(".")[:2])

        with contextlib.suppress(Exception):
            request_message = (
                f"{endpoint}python_version={python_version}"
                f"&datatype={datatype}"
                f"&ncols={ncols}"
                f"&nrows={nrows}"
                f"&ntables={ntables}"
                f"&ndocs={ndocs}"
                f"&method={method}"
                f"&os={platform.system()}"
                f"&gpu={str(gpu_present)}"
                f"&dbx={dbx}"

            )

            requests.get(request_message)

def get_datasource_info(dataset):
    from ydata.dataset import Dataset # avoiding circular import
    """
        calculate required datasource info
    """
    if isinstance(dataset, Dataset):
        nrows = dataset.nrows
        ncols = dataset.ncols
        ntables = 1
    elif isinstance(dataset, pd.DataFrame):
        nrows = len(dataset)
        ncols = len(dataset.columns)
        ntables = 1
    else:
        nrows = None
        ncols = None
        ntables = len(dataset.schema.keys())
    return nrows, ncols, ntables


class SDKLogger(logging.Logger):
    def __init__(self, name: str, level: int = logging.INFO):
        super().__init__(name, level)

    def info(self,
             datatype: str,
             method: str,
             dataset=None,
             ncols: int | None =None,
             nrows: int | None = None,  # noqa: ANN001
             ndocs: int | None = None) -> None:  # noqa: ANN001

        dbx = is_running_in_databricks()

        if datatype == 'fake':
            analytics_features(
                datatype=datatype,
                method=method,
                ncols=ncols,
                dbx=dbx
            )
        elif datatype == 'text':
            analytics_features(
                datatype=datatype,
                method=method,
                ncols=ncols,
                dbx=dbx
            )
        elif datatype == 'text':
            analytics_features(
                datatype=datatype,
                method=method,
                ncols=ncols,
                ndocs=ndocs,
                dbx=dbx
            )
        else:
            nrows, ncols, ntables = get_datasource_info(dataset)

            analytics_features(
                datatype=datatype,
                method=method,
                nrows=nrows,
                ncols=ncols,
                ntables=ntables,
                dbx=dbx
            )

        super().info(
            f"[PROFILING] Calculating profile with the following characteristics "
            f"- {datatype} | {method} | {dbx}."
        )
