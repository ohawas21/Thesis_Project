from random import sample
from string import ascii_lowercase, digits
from time import time


def generate_uuid(length=5):
    "Generates a random ID"
    s = ascii_lowercase + digits
    return "".join(sample(s, length))


def log_time_factory(logger):
    def log_time(func):
        def wrapper(*args, **kwargs):
            logger.debug(
                f"Starting {args[0].__class__.__name__}.{func.__name__} ...")

            t = time()
            result = func(*args, **kwargs)
            t = time() - t

            logger.debug(
                f"Finished {args[0].__class__.__name__}.{func.__name__} in {t} sec"
            )

            return result

        return wrapper

    return log_time
