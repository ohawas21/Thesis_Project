from __future__ import absolute_import, division, print_function

import logging
import sys


def get_logger(verbose: bool = False):
    """Get a configured logger for dat aconnectors.

    Args:
        verbose (bool, optional): if true set loglevel as debug, use info otherwhise

    Returns:
        logger
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(levelname)s: %(asctime)s %(message)s",
        level=log_level,
        stream=sys.stdout,
    )

    return logging.getLogger("dataconnectors")


logger = get_logger()
