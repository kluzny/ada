import pytest
from pathlib import Path
from unittest.mock import patch

from ada.voice import Voice


@pytest.fixture
def mock_download_voice():
    """Mock the piper download_voice function."""
    with patch("ada.voice.download_voice") as mock:
        yield mock


@pytest.fixture
def temp_voice_dir(tmp_path, monkeypatch):
    """Create a temporary voices directory for testing."""
    voice_dir = tmp_path / "voices"
    voice_dir.mkdir()
    monkeypatch.setattr(Voice, "CACHE_DIR", str(voice_dir))
    return voice_dir


def test_voices_initialization(temp_voice_dir, mock_download_voice):
    """Test Voice initialization with a voice identifier."""
    voice = Voice("en_US-amy-medium")

    assert voice.voice == "en_US-amy-medium"


def test_voices_downloads_when_missing(temp_voice_dir, mock_download_voice):
    """Test that Voice downloads the voice when files don't exist."""
    Voice("en_US-amy-medium")

    # Verify download_voice was called
    mock_download_voice.assert_called_once_with(
        "en_US-amy-medium", Path(temp_voice_dir), force_redownload=False
    )


def test_voices_skips_download_when_exists(temp_voice_dir, mock_download_voice):
    """Test that Voice skips download when voice files already exist."""
    # Create mock voice files
    model_file = temp_voice_dir / "en_US-amy-medium.onnx"
    config_file = temp_voice_dir / "en_US-amy-medium.onnx.json"
    model_file.touch()
    config_file.touch()

    Voice("en_US-amy-medium")

    # Verify download_voice was NOT called
    mock_download_voice.assert_not_called()


def test_voices_get_model_path(temp_voice_dir, mock_download_voice):
    """Test get_model_path returns correct path."""
    voice = Voice("en_US-amy-medium")

    expected_path = str(temp_voice_dir / "en_US-amy-medium.onnx")
    assert voice.get_model_path() == expected_path


def test_voices_get_config_path(temp_voice_dir, mock_download_voice):
    """Test get_config_path returns correct path."""
    voice = Voice("en_US-amy-medium")

    expected_path = str(temp_voice_dir / "en_US-amy-medium.onnx.json")
    assert voice.get_config_path() == expected_path


def test_voices_with_different_voice(temp_voice_dir, mock_download_voice):
    """Test Voice with a different voice identifier."""
    voice = Voice("en_GB-semaine-medium")

    assert voice.voice == "en_GB-semaine-medium"
    mock_download_voice.assert_called_once_with(
        "en_GB-semaine-medium", Path(temp_voice_dir), force_redownload=False
    )


def test_voices_download_error_handling(temp_voice_dir):
    """Test that Voice raises error when download fails."""
    with patch("ada.voice.download_voice") as mock_download:
        mock_download.side_effect = Exception("Download failed")

        with pytest.raises(Exception, match="Download failed"):
            Voice("en_US-amy-medium")


def test_voices_voice_exists_both_files_required(temp_voice_dir, mock_download_voice):
    """Test that both .onnx and .onnx.json files are required."""
    # Only create the model file, not the config
    model_file = temp_voice_dir / "en_US-amy-medium.onnx"
    model_file.touch()

    # This should still try to download because config file is missing
    Voice("en_US-amy-medium")
    mock_download_voice.assert_called_once()
