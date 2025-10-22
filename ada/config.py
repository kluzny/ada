import logging
import json
from typing import Any


class Config:
    DEFAULT_PATH = "config.json"

    def __init__(self, path: str | None = None):
        self.config_path: str = path or self.DEFAULT_PATH
        self.loaded: dict = self.__init__load(self.config_path)

    def __init__load(self, path: str) -> dict:
        with open(path, "r") as f:
            return json.load(f)

    def log_level(self) -> int:
        level = self.loaded["log_level"] if "log_level" in self.loaded else "WARNING"
        return getattr(logging, level)

    def record(self) -> bool:
        return "record" in self.loaded and self.loaded["record"]

    def history(self) -> bool:
        return "history" in self.loaded and self.loaded["history"]

    def voice(self) -> str | bool:
        """
        Get the TTS voice configuration.

        Returns:
            Voice model string (e.g., "en_US-amy-medium") if present and not blank,
            otherwise False
        """
        tts = self.loaded.get("tts", "")
        return tts if tts else False

    def backend(self) -> str:
        """
        Get the backend from configuration.

        Returns:
            Backend: "llama-cpp" or "ollama" (default: "llama-cpp")
        """
        return self.loaded.get("backend", "llama-cpp")

    def backend_config(self, backend: str | None = None) -> dict[str, Any]:
        """
        Get the raw backend-specific configuration.

        Args:
            backend: Optional backend name. If None, uses the configured backend.

        Returns:
            Dictionary with the raw backend configuration from the config file
        """
        backend = backend or self.backend()
        return self.__get_backend_config_for(backend)

    def __get_backends_config(self) -> dict[str, Any]:
        """Get the backends configuration object."""
        if "backends" not in self.loaded:
            raise ValueError("Missing 'backends' configuration")
        return self.loaded["backends"]

    def __get_backend_config_for(self, backend: str) -> dict[str, Any]:
        """
        Get backend configuration for a specific backend.

        Args:
            backend: The backend name (e.g., "llama-cpp", "ollama")

        Returns:
            Dictionary with the backend configuration

        Raises:
            ValueError: If the backend configuration is missing
        """
        backends = self.__get_backends_config()
        if backend not in backends:
            raise ValueError(f"Missing '{backend}' backend configuration")
        return backends[backend]


if __name__ == "__main__":
    config = Config()

    print(f"CONFIG_PATH: {config.config_path}")
    print("Loaded:")
    print(json.dumps(config.loaded, indent=4))
