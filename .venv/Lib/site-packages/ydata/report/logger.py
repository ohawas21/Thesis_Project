import logging
from os import getenv
from sys import stdout

logger = logging.getLogger("syntheticdatareport")


def symlogger_config(verbose):
    logger = logging.getLogger("Simulator")

    log_file = getenv("LOG_FILE")

    if log_file:
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            format="%(levelname)s: %(asctime)s %(message)s",
            level=log_level,
            filename=getenv("SYM_LOG_PATH"),
        )
    else:
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            format="%(levelname)s: %(asctime)s %(message)s",
            level=log_level,
            stream=stdout,
        )

    return logger
