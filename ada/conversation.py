import json
import time
import uuid

from pathlib import Path
from pydantic import BaseModel, Field

from ada.entry import Entry
from ada.formatter import block
from ada.logger import build_logger
from ada.response import Response

logger = build_logger(__name__)


class Conversation(BaseModel):
    """
    A history of Agent interactions
    """

    STORAGE_PATH: str = "conversations"

    history: list[Entry] = Field(
        default_factory=list,
        description="List of conversation entries",
    )
    record: bool = Field(
        default=False,
        description="Whether to record conversation history to JSON file",
    )
    record_path: str | None = None
    storage_path: str | None = None

    def __init__(self, **data):
        super().__init__(**data)

        if self.record:
            self.__initialize_storage_path()
            self.record_path = self.storage_path + "/" + self.__generate_file_name()

    def __initialize_storage_path(self):
        if self.storage_path is None:
            self.storage_path = self.STORAGE_PATH

        # Ensure conversations directory exists
        conversations_dir = Path(self.storage_path)
        conversations_dir.mkdir(exist_ok=True)
        logger.info(f"recording conversation to: {self.storage_path}")

    def append(self, author: str, body: str):
        entry = Entry(author=author, body=body)
        self.history.append(entry)
        if self.record:
            self.__save_record()

    def append_response(self, author: str, response: Response):
        entry = Entry(
            author=author,
            body=response.body,
            role=response.role,
            content=response.content,
        )
        self.history.append(entry)
        if self.record:
            self.__save_record()

    def clear(self):
        self.history = []
        if self.record:
            self.__remove_record()
        print(block("HISTORY CLEARED").strip())

    def messages(self) -> list[dict]:
        return [entry.message() for entry in self.history]

    def __str__(self):
        output = ""
        output += block("HISTORY START")
        for entry in self.history:
            output += str(entry) + "\n"
        output += block("HISTORY END")
        return output.strip()

    def __generate_file_name(self) -> str:
        timestamp = int(time.time())
        unique_id = str(uuid.uuid4())
        return f"{timestamp}-{unique_id}.json"

    def __save_record(self):
        """Save current history to JSON file"""
        history_data = []
        for entry in self.history:
            history_data.append(entry.model_dump())

        logger.info(f"saving to record file: {self.record_path}")
        with open(self.record_path, "w") as f:
            json.dump(history_data, f, indent=4)

    def __remove_record(self):
        """Remove the history file"""
        if self.record_path and Path(self.record_path).exists():
            logger.info(f"removing record file: {self.record_path}")
            Path(self.record_path).unlink()
