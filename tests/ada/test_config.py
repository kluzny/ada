import logging
import pytest

from ada.config import Config


EXAMPLE_CONFIG = {
    "log_level": "DEBUG",
    "record": False,
    "history": False,
    "tts": "en_US-amy-medium",
    "stt": "medium.en",
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
                }
            ],
        },
        "ollama": {
            "url": "http://localhost:11434",
            "model": "llama2",
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


def test_voice_with_valid_tts(example_config):
    """Test voice() returns the TTS voice model when configured."""
    voice = example_config.voice()
    assert voice == "en_US-amy-medium"


def test_voice_with_missing_tts():
    """Test voice() returns False when tts key is missing."""
    config = Config.__new__(Config)
    config.loaded = {"log_level": "DEBUG"}

    voice = config.voice()
    assert voice is False


def test_voice_with_blank_tts():
    """Test voice() returns False when tts value is blank."""
    config = Config.__new__(Config)
    config.loaded = {"log_level": "DEBUG", "tts": ""}

    voice = config.voice()
    assert voice is False


def test_listen_with_valid_stt(example_config):
    """Test listen() returns the STT model when configured."""
    listen_model = example_config.listen()
    assert listen_model == "medium.en"


def test_listen_with_missing_stt():
    """Test listen() returns False when stt key is missing."""
    config = Config.__new__(Config)
    config.loaded = {"log_level": "DEBUG"}

    listen_model = config.listen()
    assert listen_model is False


def test_listen_with_blank_stt():
    """Test listen() returns False when stt value is blank."""
    config = Config.__new__(Config)
    config.loaded = {"log_level": "DEBUG", "stt": ""}

    listen_model = config.listen()
    assert listen_model is False


def test_listen_with_different_model():
    """Test listen() with different STT model."""
    config = Config.__new__(Config)
    config.loaded = {"log_level": "DEBUG", "stt": "small.en"}

    listen_model = config.listen()
    assert listen_model == "small.en"
