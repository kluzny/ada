import os
import urllib.request

from ada.logger import build_logger

logger = build_logger(__name__)


class Model:
    CACHE_DIR = "models"

    url = None
    name = None
    path = None

    def __init__(self, url: str):
        self.url = url
        self.name = url.split("/")[-1]
        logger.debug(f"using {self.name}")
        self.path = os.path.join(self.CACHE_DIR, self.name)

        self.prepare()

    def prepare(self):
        os.makedirs(self.CACHE_DIR, exist_ok=True)

        if not os.path.exists(self.path):
            logger.info(f"downloading from {self.url}...")
            urllib.request.urlretrieve(self.url, self.path)
            logger.info(f"saved to {self.path}")
        else:
            logger.info(f"exists at {self.path}")
