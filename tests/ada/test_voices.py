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
    with patch("ada.voice.PiperVoice"):
        voice = Voice("en_US-amy-medium")

        assert voice.voice == "en_US-amy-medium"


def test_voices_downloads_when_missing(temp_voice_dir, mock_download_voice):
    """Test that Voice downloads the voice when files don't exist."""
    with patch("ada.voice.PiperVoice"):
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

    with patch("ada.voice.PiperVoice"):
        Voice("en_US-amy-medium")

        # Verify download_voice was NOT called
        mock_download_voice.assert_not_called()


def test_voices_get_model_path(temp_voice_dir, mock_download_voice):
    """Test get_model_path returns correct path."""
    with patch("ada.voice.PiperVoice"):
        voice = Voice("en_US-amy-medium")

        expected_path = str(temp_voice_dir / "en_US-amy-medium.onnx")
        assert voice.get_model_path() == expected_path


def test_voices_get_config_path(temp_voice_dir, mock_download_voice):
    """Test get_config_path returns correct path."""
    with patch("ada.voice.PiperVoice"):
        voice = Voice("en_US-amy-medium")

        expected_path = str(temp_voice_dir / "en_US-amy-medium.onnx.json")
        assert voice.get_config_path() == expected_path


def test_voices_with_different_voice(temp_voice_dir, mock_download_voice):
    """Test Voice with a different voice identifier."""
    with patch("ada.voice.PiperVoice"):
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

    with patch("ada.voice.PiperVoice"):
        # This should still try to download because config file is missing
        Voice("en_US-amy-medium")
        mock_download_voice.assert_called_once()


def test_voices_say_streams_audio(temp_voice_dir, mock_download_voice):
    """Test that say() streams audio from Piper to PyAudio."""
    # Create mock voice files
    model_file = temp_voice_dir / "en_US-amy-medium.onnx"
    config_file = temp_voice_dir / "en_US-amy-medium.onnx.json"
    model_file.touch()
    config_file.touch()

    with patch("ada.voice.PiperVoice") as mock_piper_voice:
        # Mock the audio chunk
        mock_chunk = type(
            "AudioChunk",
            (),
            {
                "sample_rate": 22050,
                "sample_channels": 1,
                "sample_width": 2,
                "audio_int16_bytes": b"\x00\x00",
            },
        )()

        # Mock the synthesize generator
        mock_piper_instance = mock_piper_voice.load.return_value
        mock_piper_instance.synthesize.return_value = [mock_chunk]

        with patch("ada.voice.pyaudio.PyAudio") as mock_pyaudio:
            # Setup PyAudio mock
            mock_p = mock_pyaudio.return_value
            mock_stream = mock_p.open.return_value
            mock_p.get_format_from_width.return_value = 8

            voice = Voice("en_US-amy-medium")
            voice.say("Hello world")

            # Verify PyAudio was used correctly
            mock_p.open.assert_called_once()
            mock_stream.write.assert_called_once_with(b"\x00\x00")
            mock_stream.stop_stream.assert_called_once()
            mock_stream.close.assert_called_once()
            mock_p.terminate.assert_called_once()


def test_voices_say_handles_multiple_chunks(temp_voice_dir, mock_download_voice):
    """Test that say() handles multiple audio chunks correctly."""
    # Create mock voice files
    model_file = temp_voice_dir / "en_US-amy-medium.onnx"
    config_file = temp_voice_dir / "en_US-amy-medium.onnx.json"
    model_file.touch()
    config_file.touch()

    with patch("ada.voice.PiperVoice") as mock_piper_voice:
        # Mock multiple audio chunks
        mock_chunk1 = type(
            "AudioChunk",
            (),
            {
                "sample_rate": 22050,
                "sample_channels": 1,
                "sample_width": 2,
                "audio_int16_bytes": b"\x00\x01",
            },
        )()
        mock_chunk2 = type(
            "AudioChunk",
            (),
            {
                "sample_rate": 22050,
                "sample_channels": 1,
                "sample_width": 2,
                "audio_int16_bytes": b"\x02\x03",
            },
        )()

        # Mock the synthesize generator
        mock_piper_instance = mock_piper_voice.load.return_value
        mock_piper_instance.synthesize.return_value = [mock_chunk1, mock_chunk2]

        with patch("ada.voice.pyaudio.PyAudio") as mock_pyaudio:
            # Setup PyAudio mock
            mock_p = mock_pyaudio.return_value
            mock_stream = mock_p.open.return_value
            mock_p.get_format_from_width.return_value = 8

            voice = Voice("en_US-amy-medium")
            voice.say("Hello world, this is a longer message")

            # Verify that write was called for both chunks
            assert mock_stream.write.call_count == 2
            mock_stream.write.assert_any_call(b"\x00\x01")
            mock_stream.write.assert_any_call(b"\x02\x03")


def test_voices_say_error_handling(temp_voice_dir, mock_download_voice):
    """Test that say() handles errors gracefully."""
    # Create mock voice files
    model_file = temp_voice_dir / "en_US-amy-medium.onnx"
    config_file = temp_voice_dir / "en_US-amy-medium.onnx.json"
    model_file.touch()
    config_file.touch()

    with patch("ada.voice.PiperVoice") as mock_piper_voice:
        # Mock the synthesize to raise an error
        mock_piper_instance = mock_piper_voice.load.return_value
        mock_piper_instance.synthesize.side_effect = Exception("Synthesis failed")

        with patch("ada.voice.pyaudio.PyAudio") as mock_pyaudio:
            mock_p = mock_pyaudio.return_value
            mock_p.open.return_value

            voice = Voice("en_US-amy-medium")

            # Verify that the exception is raised and resources are cleaned up
            with pytest.raises(Exception, match="Synthesis failed"):
                voice.say("Hello world")

            # Verify cleanup still happens
            mock_p.terminate.assert_called_once()


def test_voices_say_stream_cleanup_on_error(temp_voice_dir, mock_download_voice):
    """Test that stream is properly cleaned up even if write fails."""
    # Create mock voice files
    model_file = temp_voice_dir / "en_US-amy-medium.onnx"
    config_file = temp_voice_dir / "en_US-amy-medium.onnx.json"
    model_file.touch()
    config_file.touch()

    with patch("ada.voice.PiperVoice") as mock_piper_voice:
        # Mock the audio chunk
        mock_chunk = type(
            "AudioChunk",
            (),
            {
                "sample_rate": 22050,
                "sample_channels": 1,
                "sample_width": 2,
                "audio_int16_bytes": b"\x00\x00",
            },
        )()

        # Mock the synthesize generator
        mock_piper_instance = mock_piper_voice.load.return_value
        mock_piper_instance.synthesize.return_value = [mock_chunk]

        with patch("ada.voice.pyaudio.PyAudio") as mock_pyaudio:
            # Setup PyAudio mock with stream.write error
            mock_p = mock_pyaudio.return_value
            mock_stream = mock_p.open.return_value
            mock_stream.write.side_effect = Exception("Write failed")
            mock_p.get_format_from_width.return_value = 8

            voice = Voice("en_US-amy-medium")

            # Verify that the exception is raised
            with pytest.raises(Exception, match="Write failed"):
                voice.say("Hello world")

            # Verify stream cleanup happens
            mock_stream.stop_stream.assert_called_once()
            mock_stream.close.assert_called_once()
            mock_p.terminate.assert_called_once()
