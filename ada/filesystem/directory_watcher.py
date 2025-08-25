from asyncio import Queue, sleep, AbstractEventLoop
from pathlib import Path
from watchdog.observers import Observer

from ada.logger import build_logger
from ada.filesystem.async_file_watcher import AsyncFileWatcher

logger = build_logger(__name__)


class DirectoryWatcher:
    """Wrapper around watchdog observer with async integration."""

    def __init__(self, path: Path, loop: AbstractEventLoop, queue: Queue):
        self.path = path
        self.loop = loop
        self.queue = queue
        self.observer = Observer()
        self.handler = AsyncFileWatcher(loop, queue)

    async def start(self):
        self.observer.schedule(self.handler, str(self.path), recursive=True)
        self.observer.start()
        logger.info(f"started watching directory tree: {self.path.resolve()}")

        try:
            while self.observer.is_alive():
                await sleep(1)
        finally:
            await self.stop()

    async def stop(self):
        if self.observer.is_alive():
            logger.info("stopping directory watcher...")
            self.observer.stop()
            self.observer.join()
            logger.info("watcher stopped.")
