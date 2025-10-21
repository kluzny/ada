import logging
import pytest

from ada.config import Config


def tests_config():
    Config()


def test_config_load_and_access():
    config = Config("tests/fixtures/config/example.json")

    assert config.log_level() == logging.DEBUG
    assert not config.record()
    assert not config.history()


def test_config_minimum_viable_defaults():
    config = Config("tests/fixtures/config/minimum.json")
    assert config.log_level() == logging.WARNING
    assert not config.record()
    assert not config.history()


def test_backend_config_without_arguments():
    """Test backend_config returns the configured backend when no argument is provided."""
    config = Config("tests/fixtures/config/example.json")

    # The example.json has "backend": "llama-cpp"
    backend_config = config.backend_config()

    # Should return llama-cpp configuration
    assert "model" in backend_config
    assert "models" in backend_config
    assert backend_config["model"] == "tiny-llm"
    assert isinstance(backend_config["models"], list)


def test_backend_config_with_llama_cpp_argument():
    """Test backend_config with explicit 'llama-cpp' argument."""
    config = Config("tests/fixtures/config/example.json")

    backend_config = config.backend_config("llama-cpp")

    # Should return llama-cpp configuration
    assert "model" in backend_config
    assert "models" in backend_config
    assert backend_config["model"] == "tiny-llm"
    assert isinstance(backend_config["models"], list)
    assert len(backend_config["models"]) > 0
    assert backend_config["models"][0]["name"] == "tiny-llm"


def test_backend_config_with_ollama_argument():
    """Test backend_config with explicit 'ollama' argument."""
    config = Config("tests/fixtures/config/example.json")

    backend_config = config.backend_config("ollama")

    # Should return ollama configuration
    assert "model" in backend_config
    assert "url" in backend_config
    assert backend_config["model"] == "llama2"
    assert backend_config["url"] == "http://localhost:11434"


def test_backend_config_with_invalid_backend():
    """Test backend_config raises error with invalid backend name."""
    config = Config("tests/fixtures/config/example.json")

    with pytest.raises(ValueError, match="Unknown backend: invalid-backend"):
        config.backend_config("invalid-backend")
