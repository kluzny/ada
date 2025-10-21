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

    def __find_llama_cpp_model(self, model_name: str) -> dict[str, Any]:
        """Find a model definition by name in llama-cpp models list."""
        llama_config = self.__get_backend_config_for("llama-cpp")
        if "models" not in llama_config:
            raise ValueError("Missing 'models' array in llama-cpp configuration")

        for model in llama_config["models"]:
            if model.get("name") == model_name:
                return model

        raise ValueError(f"Model '{model_name}' not found in llama-cpp models list")

    def model_tokens(self) -> int:
        """Get the token context size for the current backend."""
        backend = self.backend()

        if backend == "llama-cpp":
            llama_config = self.__get_backend_config_for("llama-cpp")
            model_name = llama_config.get("model")
            if not model_name:
                raise ValueError("Missing 'model' in llama-cpp configuration")

            model_def = self.__find_llama_cpp_model(model_name)
            return model_def.get("tokens", 2048)
        elif backend == "ollama":
            ollama_config = self.__get_backend_config_for("ollama")
            return ollama_config.get("tokens", 2048)
        else:
            return 2048


if __name__ == "__main__":
    config = Config()

    print(f"CONFIG_PATH: {config.config_path}")
    print("Loaded:")
    print(json.dumps(config.loaded, indent=4))
