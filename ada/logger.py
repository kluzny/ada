import logging
import sys
import os

from ada.config import Config

FORMAT = "%(asctime)s|%(name)s|%(levelname)s|%(message)s"
ERROR_THRESHOLD = logging.ERROR

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "ada.log")
LOG_STD: bool = os.environ.get("LOG_STD") is not None


def build_logger(name: str):
    config = Config()

    logger = logging.getLogger(name)
    logger.setLevel(config.log_level())

    if logger.hasHandlers():
        return logger

    formatter = logging.Formatter(FORMAT)

    if LOG_STD:
        default_handler = logging.StreamHandler(sys.stdout)
        error_handler = logging.StreamHandler(sys.stderr)
    else:
        os.makedirs(LOG_DIR, exist_ok=True)
        default_handler = logging.FileHandler(LOG_FILE)
        error_handler = logging.FileHandler(LOG_FILE)

    default_handler.setLevel(logging.NOTSET)
    default_handler.addFilter(lambda record: record.levelno < ERROR_THRESHOLD)
    default_handler.setFormatter(formatter)

    error_handler.setLevel(ERROR_THRESHOLD)
    error_handler.setFormatter(formatter)

    logger.addHandler(default_handler)
    logger.addHandler(error_handler)

    return logger
