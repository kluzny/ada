import os

from asyncio import Queue, AbstractEventLoop
from functools import cached_property
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
    WRAPPER_TAG = "memory"
    START_TAG = f"<{WRAPPER_TAG}>"
    END_TAG = f"</{WRAPPER_TAG}>"
    INSTRUCTION = (
        f"IMPORTANT: Additional instructions are wrapped with {START_TAG}{END_TAG}"
    )

    def __init__(self, name: str, description: str = "", prompt: str = ""):
        self.name = name
        self.description = description
        self.prompt = prompt
        self.watcher = None

    def clear_cached_memories(self) -> None:
        if hasattr(self, "_cached_memories"):
            del self._cached_memories

    def get_prompt(self) -> str:
        prompts = [self.prompt]
        if len(self._cached_memories) > 0:
            prompts.append("\n" + self.INSTRUCTION)
            prompts.append(self._cached_memories)

        return "\n".join(prompts)

    async def watch(self, loop: AbstractEventLoop, queue: Queue) -> None:
        path = self._memory_path()
        path.mkdir(exist_ok=True)

        self.watcher = DirectoryWatcher(path, loop, queue)
        await self.watcher.start()

    def unwatch(self) -> None:
        if self.watcher is not None:
            self.watcher.stop()

    def _memory_path(self) -> Path:
        return Path(os.path.join(self.MEMORIES_PATH, self.name))

    def _get_memory_files(self) -> list[str]:
        """stringified paths, to allow for alpha sorting"""
        return sorted([str(p) for p in self._memory_path().rglob("*") if p.is_file()])

    def _commands(self) -> list[str]:
        memories = []
        for path in self._get_memory_files():
            with open(path, "r") as memory:
                contents = memory.read()

                if len(contents) > 0:
                    if contents[-1] == "\n":
                        padding = ""
                    else:
                        padding = "\n"

                    memories.append(
                        f"{self.START_TAG}\n{contents}{padding}{self.END_TAG}\n"
                    )
        return memories

    @cached_property
    def _cached_memories(self) -> str:
        return "\n".join(self._commands())

    def __str__(self) -> str:
        return f"{self.name}: {self.description}"

    def __repr__(self) -> str:
        return f"Persona(name='{self.name}', description='{self.description}', prompt='{self.prompt}')"
