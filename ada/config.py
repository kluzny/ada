import logging
import json


class Config:
    CONFIG_PATH = "config.json"

    loaded = None

    def __init__(self):
        self.loaded = self.load(self.CONFIG_PATH)

    def load(self, path: str):
        with open(path, "r") as f:
            return json.load(f)

    def model_url(self) -> str:
        return self.loaded["model_url"]

    def log_level(self) -> int:
        level = self.loaded["log_level"]
        return getattr(logging, level)


if __name__ == "__main__":
    config = Config()

    print(f"CONFIG_PATH: {config.CONFIG_PATH}")
    print("Loaded:")
    print(json.dumps(config.loaded, indent=4))
