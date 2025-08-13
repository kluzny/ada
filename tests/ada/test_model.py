from ada.model import Model
from ada.config import Config

config = Config("tests/fixtures/config/example.json")


def test_model():
    Model(url=config.model_url())
