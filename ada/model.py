import os
import urllib.request

from prompt_toolkit.shortcuts import ProgressBar

from ada.logger import build_logger
from ada.constants import ARTIFACT_DIR

logger = build_logger(__name__)


class Model:
    CACHE_DIR = ARTIFACT_DIR / "llms"
    CHUNK_SIZE = 1024  # 1kb

    def __init__(self, url: str):
        self.url: str = url
        self.name: str = url.split("/")[-1]
        logger.debug(f"using {self.name}")
        self.path: str = os.path.join(self.CACHE_DIR, self.name)

        self.__prepare()

    def __prepare(self) -> None:
        os.makedirs(self.CACHE_DIR, exist_ok=True)

        if not os.path.exists(self.path):
            logger.info(f"downloading from {self.url}...")
            self.__download()
            logger.info(f"saved to {self.path}")
        else:
            logger.info(f"exists at {self.path}")

    def __download(self) -> None:
        with urllib.request.urlopen(self.url) as response:
            content_length = int(response.getheader("Content-Length", 0))
            total = round(content_length / self.CHUNK_SIZE)

            with open(self.path, "wb") as f:
                # wrap an iterable that yields chunk sizes
                def download_iterable():
                    while True:
                        chunk = response.read(self.CHUNK_SIZE)
                        if not chunk:
                            break
                        f.write(chunk)
                        nonlocal_downloaded[0] += len(chunk)
                        yield nonlocal_downloaded[0]

                nonlocal_downloaded = [0]  # hacky mutable accumulator

                with ProgressBar() as pb:
                    for _done in pb(download_iterable(), total=total, label=self.path):
                        pass
