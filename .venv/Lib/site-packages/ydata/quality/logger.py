import logging
from os import getenv
from sys import stdout


def qualitylogger_config(verbose):
    log_level = logging.DEBUG if verbose else logging.INFO

    log_file = getenv("LOG_FILE")
    if log_file:
        logging.basicConfig(
            format="%(levelname)s: %(asctime)s %(message)s",
            level=log_level,
            filename=getenv("QUALITY_LOG_PATH"),
        )
    else:
        logging.basicConfig(
            format="%(levelname)s: %(asctime)s %(message)s",
            level=log_level,
            stream=stdout,
        )

    logger = logging.getLogger("Quality")
    return logger
