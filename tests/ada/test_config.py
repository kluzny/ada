import logging
from ada.config import Config


def tests_config():
    Config()


def test_config_load_and_access():
    config = Config("tests/fixtures/config/example.json")
    assert config.log_level() == logging.DEBUG
    assert config.model_url() == "https://example.com/model.gguf"
    assert config.model_tokens() == 2048
