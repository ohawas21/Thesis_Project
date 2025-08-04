"""Connectors utility functions."""
from __future__ import absolute_import, division, print_function

import datetime
from contextlib import contextmanager
from decimal import Decimal
from os import environ, makedirs
from os import path as os_path
from os import walk

from ydata.connectors.exceptions import DataConnectorsException
from ydata.connectors.logger import logger


def get_from_env(keys):
    """Returns an environment variable from one of the list of keys.

    Args:
        keys: list(str). list of keys to check in the environment
    Returns:
        str | None
    """
    keys = keys or []
    if not isinstance(keys, (list, tuple)):
        keys = [keys]
    for key in keys:
        value = environ.get(key)
        if value:
            if value.lower() == "true":
                return True
            if value.lower() == "false":
                return False
            return value
        # Prepend YDATA
        key = "YDATA_{}".format(key)
        value = environ.get(key)
        if value:
            return value
    return None


def is_protected_type(obj):
    """A check for preserving a type as-is when passed to
    force_text(strings_only=True)."""
    return isinstance(
        obj,
        (
            type(None),
            int,
            float,
            Decimal,
            datetime.datetime,
            datetime.date,
            datetime.time,
        ),
    )


@contextmanager
def get_files_in_current_directory(path):
    """Gets all the files under a certain path.

    Args:
        path: `str`. The path to traverse for collecting files.
    Returns:
         list of files collected under the path.
    """
    result_files = []

    for root, dirs, files in walk(path):
        logger.debug("Root:%s, Dirs:%s", root, dirs)

        for file_name in files:
            result_files.append(os_path.join(root, file_name))

    yield result_files


def append_basename(path, filename):
    """Adds the basename of the filename to the path.

    Args:
        path: `str`. The path to append the basename to.
        filename: `str`. The filename to extract the base name from.
    Returns:
         str
    """
    return os_path.join(path, os_path.basename(filename))


def check_dirname_exists(path, is_dir=False):
    if not is_dir:
        path = os_path.dirname(os_path.abspath(path))
    if not os_path.isdir(path):
        raise DataConnectorsException(
            "The parent path is not a directory {}".format(path)
        )


def create_tmp():
    base_path = os_path.join("/tmp", ".ydata")
    if not os_path.exists(base_path):
        try:
            makedirs(base_path)
        except OSError:
            # Except permission denied and potential race conditions
            # in multi-threaded environments.
            logger.warning("Could not create config directory `%s`", base_path)
    return base_path
