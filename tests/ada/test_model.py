from unittest.mock import patch
from ada.model import Model

TEST_MODEL_URL = "https://example.com/model.gguf"

def test_model():
    with patch("ada.model.Model._Model__prepare") as _prepare:
        model = Model(url=TEST_MODEL_URL)

        assert model.url == TEST_MODEL_URL
        assert model.name == "model.gguf"
        assert model.path == "models/model.gguf"

def test_model_init_calls_prepare():
    with patch("ada.model.Model._Model__prepare") as prepare:
        Model(url=TEST_MODEL_URL)
        prepare.assert_called_once()

def test_model_init_file_exists():
    with patch("ada.model.Model._Model__download") as download, \
          patch("os.path.exists", return_value=True) as _exists:
        Model(url=TEST_MODEL_URL)
        download.assert_not_called()

def test_model_init_file_does_not_exist():
    with patch("ada.model.Model._Model__download") as download, \
          patch("os.path.exists", return_value=False) as _exists:
        Model(url=TEST_MODEL_URL)
        download.assert_called_once()