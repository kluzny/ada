import logging
import json
from typing import Any


class Config:
    DEFAULT_PATH = "config.json"

    def __init__(self, path: str | None = None):
        self.config_path: str = path or self.DEFAULT_PATH
        self.loaded: dict = self.__load(self.config_path)

    def __load(self, path: str) -> dict:
        with open(path, "r") as f:
            return json.load(f)

    def log_level(self) -> int:
        level = self.loaded["log_level"] if "log_level" in self.loaded else "WARNING"
        return getattr(logging, level)

    def record(self) -> bool:
        return "record" in self.loaded and self.loaded["record"]

    def history(self) -> bool:
        return "history" in self.loaded and self.loaded["history"]

    def backend_type(self) -> str:
        """
        Get the backend type from configuration.

        Returns:
            Backend type: "llama-cpp" or "ollama" (default: "llama-cpp")
        """
        return self.loaded.get("backend", "llama-cpp")

    def __get_backends_config(self) -> dict[str, Any]:
        """Get the backends configuration object."""
        if "backends" not in self.loaded:
            raise ValueError("Missing 'backends' configuration")
        return self.loaded["backends"]

    def __get_llama_cpp_config(self) -> dict[str, Any]:
        """Get llama-cpp backend configuration."""
        backends = self.__get_backends_config()
        if "llama-cpp" not in backends:
            raise ValueError("Missing 'llama-cpp' backend configuration")
        return backends["llama-cpp"]

    def __get_ollama_config(self) -> dict[str, Any]:
        """Get ollama backend configuration."""
        backends = self.__get_backends_config()
        if "ollama" not in backends:
            raise ValueError("Missing 'ollama' backend configuration")
        return backends["ollama"]

    def __find_llama_cpp_model(self, model_name: str) -> dict[str, Any]:
        """Find a model definition by name in llama-cpp models list."""
        llama_config = self.__get_llama_cpp_config()
        if "models" not in llama_config:
            raise ValueError("Missing 'models' array in llama-cpp configuration")

        for model in llama_config["models"]:
            if model.get("name") == model_name:
                return model

        raise ValueError(f"Model '{model_name}' not found in llama-cpp models list")

    def model_url(self) -> str:
        """Get the model URL for llama-cpp backend."""
        llama_config = self.__get_llama_cpp_config()
        model_name = llama_config.get("model")
        if not model_name:
            raise ValueError("Missing 'model' in llama-cpp configuration")

        model_def = self.__find_llama_cpp_model(model_name)
        return model_def["url"]

    def model_tokens(self) -> int:
        """Get the token context size for the current backend."""
        backend_type = self.backend_type()

        if backend_type == "llama-cpp":
            llama_config = self.__get_llama_cpp_config()
            model_name = llama_config.get("model")
            if not model_name:
                raise ValueError("Missing 'model' in llama-cpp configuration")

            model_def = self.__find_llama_cpp_model(model_name)
            return model_def.get("tokens", 2048)
        elif backend_type == "ollama":
            ollama_config = self.__get_ollama_config()
            return ollama_config.get("tokens", 2048)
        else:
            return 2048

    def backend_config(self) -> dict[str, Any]:
        """
        Get backend-specific configuration.

        Returns:
            Dictionary with backend configuration parameters
        """
        backend_type = self.backend_type()

        if backend_type == "llama-cpp":
            llama_config = self.__get_llama_cpp_config()
            model_name = llama_config.get("model")
            if not model_name:
                raise ValueError("Missing 'model' in llama-cpp configuration")

            model_def = self.__find_llama_cpp_model(model_name)

            return {
                "model_name": model_name,
                "model_url": model_def["url"],
                "n_ctx": model_def.get("tokens", 2048),
                "n_threads": llama_config.get("threads", 4),
                "verbose": llama_config.get("verbose", False),
            }
        elif backend_type == "ollama":
            ollama_config = self.__get_ollama_config()
            return {
                "model": ollama_config.get("model", "llama2"),
                "host": ollama_config.get("url", "http://localhost:11434"),
            }
        else:
            raise ValueError(f"Unknown backend type: {backend_type}")


if __name__ == "__main__":
    config = Config()

    print(f"CONFIG_PATH: {config.config_path}")
    print("Loaded:")
    print(json.dumps(config.loaded, indent=4))
