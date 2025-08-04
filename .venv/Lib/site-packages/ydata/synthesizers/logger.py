from __future__ import absolute_import, division, print_function

import logging
import sys
from os import getenv


def synthlogger_config(verbose):
    logger = logging.getLogger("Synthesizer")

    log_file = getenv("LOG_FILE")

    if log_file:
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            format="%(levelname)s: %(asctime)s %(message)s",
            level=log_level,
            filename=getenv("SYNTH_LOG_PATH"),
        )
    else:
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            format="%(levelname)s: %(asctime)s %(message)s",
            level=log_level,
            stream=sys.stdout,
        )

    return logger
