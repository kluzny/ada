import logging
import pytest

from ada.config import Config


EXAMPLE_CONFIG = {
    "log_level": "DEBUG",
    "record": False,
    "history": False,
    "backend": "llama-cpp",
    "backends": {
        "llama-cpp": {
            "model": "tiny-llm",
            "threads": 4,
            "verbose": False,
            "models": [
                {
                    "name": "tiny-llm",
                    "url": "https://huggingface.co/mradermacher/Tiny-LLM-GGUF/resolve/main/Tiny-LLM.IQ4_XS.gguf",
                    "tokens": 1024,
                }
            ],
        },
        "ollama": {
            "url": "http://localhost:11434",
            "model": "llama2",
            "tokens": 1024,
        },
    },
}


@pytest.fixture
def example_config():
    """Create a Config instance with mocked loaded data."""
    config = Config.__new__(Config)
    config.loaded = EXAMPLE_CONFIG
    return config


def tests_config():
    Config()


def test_config_load_and_access_example():
    config = Config("config.json.example")

    assert config.log_level() == logging.DEBUG
    assert config.record()
    assert config.history()


def test_backend_config_without_arguments(example_config):
    """Test backend_config returns the configured backend when no argument is provided."""
    # The mock config has "backend": "llama-cpp"
    backend_config = example_config.backend_config()

    # Should return llama-cpp configuration
    assert "model" in backend_config
    assert "models" in backend_config
    assert backend_config["model"] == "tiny-llm"
    assert isinstance(backend_config["models"], list)


def test_backend_config_with_llama_cpp_argument(example_config):
    """Test backend_config with explicit 'llama-cpp' argument."""
    backend_config = example_config.backend_config("llama-cpp")

    # Should return llama-cpp configuration
    assert "model" in backend_config
    assert "models" in backend_config
    assert backend_config["model"] == "tiny-llm"
    assert isinstance(backend_config["models"], list)
    assert len(backend_config["models"]) > 0
    assert backend_config["models"][0]["name"] == "tiny-llm"


def test_backend_config_with_ollama_argument(example_config):
    """Test backend_config with explicit 'ollama' argument."""
    backend_config = example_config.backend_config("ollama")

    # Should return ollama configuration
    assert "model" in backend_config
    assert "url" in backend_config
    assert backend_config["model"] == "llama2"
    assert backend_config["url"] == "http://localhost:11434"


def test_backend_config_with_invalid_backend(example_config):
    """Test backend_config raises error with invalid backend name."""
    with pytest.raises(
        ValueError, match="Missing 'invalid-backend' backend configuration"
    ):
        example_config.backend_config("invalid-backend")


def test_backend_config_missing_backends_section():
    """Test that missing backends section raises appropriate error."""
    # Create config without backends section
    config = Config.__new__(Config)
    config.loaded = {"log_level": "DEBUG", "backend": "llama-cpp"}

    with pytest.raises(ValueError, match="Missing 'backends' configuration"):
        config.backend_config()
