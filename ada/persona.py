import os

from asyncio import Queue, AbstractEventLoop
from pathlib import Path

from ada.logger import build_logger
from ada.filesystem.directory_watcher import DirectoryWatcher

logger = build_logger(__name__)


class Persona:
    """
    A persona represents a specific AI assistant personality and behavior.
    Each persona has a name, a description, and a system prompt that defines its characteristics.
    """

    MEMORIES_PATH = "memories"

    def __init__(self, name: str, description: str = "", prompt: str = ""):
        self.name = name
        self.description = description
        self.prompt = prompt
        self.watcher = None

    def _memory_path(self) -> Path:
        return Path(os.path.join(self.MEMORIES_PATH, self.name))

    async def watch(self, loop: AbstractEventLoop, queue: Queue) -> None:
        path = self._memory_path()
        path.mkdir(exist_ok=True)

        self.watcher = DirectoryWatcher(path, loop, queue)
        await self.watcher.start()

    async def unwatch(self) -> None:
        if self.watcher is not None:
            await self.watcher.stop()

    def __str__(self) -> str:
        return f"{self.name}: {self.description}"

    def __repr__(self) -> str:
        return f"Persona(name='{self.name}', description='{self.description}', prompt='{self.prompt}')"
