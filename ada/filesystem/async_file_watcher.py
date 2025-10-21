from asyncio import run_coroutine_threadsafe, Queue, AbstractEventLoop
from watchdog.events import FileSystemEventHandler, FileSystemEvent

from ada.logger import build_logger

logger = build_logger(__name__)


class AsyncFileWatcher(FileSystemEventHandler):
    def __init__(self, loop: AbstractEventLoop, queue: Queue) -> None:
        self.loop = loop
        self.queue = queue

        super().__init__()

    def _put_event(self, event_type: str, event: FileSystemEvent) -> None:
        if event.is_directory:
            return  # Ignore directory events

        logger.info(f"{event_type}:{event.src_path}")

        run_coroutine_threadsafe(
            self.queue.put(item=(event_type, event.src_path)), self.loop
        )

    def on_created(self, event: FileSystemEvent) -> None:
        self._put_event("created", event)

    def on_modified(self, event: FileSystemEvent) -> None:
        self._put_event("modified", event)

    def on_deleted(self, event: FileSystemEvent) -> None:
        self._put_event("deleted", event)
