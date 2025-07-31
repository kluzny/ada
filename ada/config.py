import logging
import json


class Config:
    CONFIG_PATH = "config.json"

    loaded: dict

    def __init__(self):
        self.loaded = self.load(self.CONFIG_PATH)

    def load(self, path: str):
        with open(path, "r") as f:
            return json.load(f)

    def log_level(self) -> int:
        level = self.loaded["log_level"]
        return getattr(logging, level)

    def model(self) -> dict:
        return self.loaded["model"]

    def model_url(self) -> str:
        return self.model()["url"]

    def model_tokens(self) -> int:
        return self.model()["tokens"]


if __name__ == "__main__":
    config = Config()

    print(f"CONFIG_PATH: {config.CONFIG_PATH}")
    print("Loaded:")
    print(json.dumps(config.loaded, indent=4))
