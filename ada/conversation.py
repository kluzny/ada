import json
import time
import uuid

from pathlib import Path
from typing import List
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

    history: List[Entry] = Field(
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
            self.__initialize_record_path()
            logger.info(f"Recording conversation to {self.record_path}")

    def __initialize_storage_path(self):
        if self.storage_path is None:
            self.storage_path = self.STORAGE_PATH

        # Ensure conversations directory exists
        conversations_dir = Path(self.storage_path)
        conversations_dir.mkdir(exist_ok=True)

    def __initialize_record_path(self):
        self.record_path = self.storage_path + "/" + self.__generate_file_name()

        # TODO: need a test for this and should probably just call __save_to_json()
        with open(self.record_path, "w") as f:
            json.dump([], f, indent=4)

    def __generate_file_name(self) -> str:
        timestamp = int(time.time())
        unique_id = str(uuid.uuid4())
        return f"{timestamp}-{unique_id}.json"

    def __save_to_json(self):
        """Save current history to JSON file"""
        history_data = []
        for entry in self.history:
            history_data.append(entry.model_dump())

        with open(self.record_path, "w") as f:
            json.dump(history_data, f, indent=4)

    def append(self, author: str, body: str):
        entry = Entry(author=author, body=body)
        self.history.append(entry)
        if self.record:
            self.__save_to_json()

    def append_response(self, author: str, response: Response):
        entry = Entry(
            author=author,
            body=response.body,
            role=response.role,
            content=response.content,
        )
        self.history.append(entry)
        if self.record:
            self.__save_to_json()

    def clear(self):
        self.history = []
        if self.record:
            self.__save_to_json()
        print(block("HISTORY CLEARED").strip())

    def messages(self) -> List[dict]:
        return [entry.message() for entry in self.history]

    def __str__(self):
        output = ""
        output += block("HISTORY START")
        for entry in self.history:
            output += str(entry) + "\n"
        output += block("HISTORY END")
        return output.strip()
