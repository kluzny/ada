import logging
from ada.config import Config


def tests_config():
    Config()


def test_config_load_and_access():
    config = Config("tests/fixtures/config/example.json")

    assert config.log_level() == logging.DEBUG
    assert not config.record()
    assert not config.history()
    assert (
        config.model_url()
        == "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q8_0.gguf"
    )
    assert config.model_tokens() == 2048


def test_config_minimum_viable_defaults():
    config = Config("tests/fixtures/config/minimum.json")
    assert config.log_level() == logging.WARNING
    assert not config.record()
    assert not config.history()
