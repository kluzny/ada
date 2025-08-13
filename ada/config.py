import logging
import json


class Config:
    DEFAULT_PATH = "config.json"

    def __init__(self, path: str = None):
        self.config_path = path or self.DEFAULT_PATH
        self.loaded = self.__load(self.config_path)

    def __load(self, path: str):
        with open(path, "r") as f:
            return json.load(f)

    def log_level(self) -> int:
        level = self.loaded["log_level"] if "log_level" in self.loaded else "WARNING"
        return getattr(logging, level)

    def record(self) -> bool:
        return "record" in self.loaded and self.loaded["record"]

    def history(self) -> bool:
        return "history" in self.loaded and self.loaded["history"]

    def __model(self) -> dict:
        return self.loaded["model"]

    def model_url(self) -> str:
        return self.__model()["url"]

    def model_tokens(self) -> int:
        return self.__model()["tokens"]


if __name__ == "__main__":
    config = Config()

    print(f"CONFIG_PATH: {config.config_path}")
    print("Loaded:")
    print(json.dumps(config.loaded, indent=4))
