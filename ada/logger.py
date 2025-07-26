import logging
import sys

from ada.config import Config

FORMAT = "%(asctime)s|%(name)s|%(levelname)s|%(message)s"
ERROR_THRESHOLD = logging.ERROR


def build_logger(name: str):
    config = Config()

    logger = logging.getLogger(name)
    logger.setLevel(config.log_level())

    if logger.hasHandlers():
        return logger

    formatter = logging.Formatter(FORMAT)

    default_handler = logging.StreamHandler(sys.stdout)
    default_handler.setLevel(logging.NOTSET)
    default_handler.addFilter(lambda record: record.levelno < ERROR_THRESHOLD)
    default_handler.setFormatter(formatter)

    error_handler = logging.StreamHandler(sys.stderr)
    error_handler.setLevel(ERROR_THRESHOLD)
    error_handler.setFormatter(formatter)

    logger.addHandler(default_handler)
    logger.addHandler(error_handler)

    return logger
