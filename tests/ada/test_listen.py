import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from ada.listen import Listen


@pytest.fixture
def mock_whisper_model():
    """Mock the WhisperModel for testing."""
    with patch("ada.listen.WhisperModel") as mock:
        yield mock


@pytest.fixture
def temp_model_dir(tmp_path, monkeypatch):
    """Create a temporary models directory for testing."""
    model_dir = tmp_path / "models" / "whisper"
    model_dir.mkdir(parents=True)
    monkeypatch.setattr(Listen, "CACHE_DIR", model_dir)
    return model_dir


@pytest.fixture
def mock_pyaudio():
    """Mock PyAudio for testing."""
    with patch("ada.listen.pyaudio.PyAudio") as mock:
        yield mock


def test_listen_initialization(temp_model_dir, mock_whisper_model):
    """Test Listen initialization with a model identifier."""
    listen = Listen("medium.en")

    assert listen.model_name == "medium.en"
    mock_whisper_model.assert_called_once()


def test_listen_initialization_default_model(temp_model_dir, mock_whisper_model):
    """Test Listen initialization with default model."""
    listen = Listen()

    assert listen.model_name == "medium.en"
    mock_whisper_model.assert_called_once()


def test_listen_initialization_custom_model(temp_model_dir, mock_whisper_model):
    """Test Listen initialization with custom model."""
    listen = Listen("small.en")

    assert listen.model_name == "small.en"
    mock_whisper_model.assert_called_once()


def test_listen_cache_dir_creation(temp_model_dir, mock_whisper_model):
    """Test that cache directory is created during initialization."""
    Listen()

    assert temp_model_dir.exists()


def test_listen_capture_audio_basic(temp_model_dir, mock_whisper_model, mock_pyaudio):
    """Test basic audio capture from microphone."""
    # Mock the audio stream
    mock_p = mock_pyaudio.return_value
    mock_stream = MagicMock()
    mock_p.open.return_value = mock_stream

    # Mock audio data - create float32 samples
    audio_chunk = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32).tobytes()
    mock_stream.read.return_value = audio_chunk

    # Mock the transcribe response
    mock_segment = MagicMock()
    mock_segment.text = "hello world"
    mock_whisper_instance = mock_whisper_model.return_value
    mock_whisper_instance.transcribe.return_value = ([mock_segment], MagicMock())

    listen = Listen()
    result = listen.listen(duration=0.1)

    assert result == "hello world"
    mock_stream.stop_stream.assert_called_once()
    mock_stream.close.assert_called_once()
    mock_p.terminate.assert_called_once()


def test_listen_capture_multiple_frames(
    temp_model_dir, mock_whisper_model, mock_pyaudio
):
    """Test audio capture with multiple frames."""
    mock_p = mock_pyaudio.return_value
    mock_stream = MagicMock()
    mock_p.open.return_value = mock_stream

    # Create multiple audio chunks
    audio_chunk = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32).tobytes()
    mock_stream.read.return_value = audio_chunk

    # Mock the transcribe response
    mock_segment = MagicMock()
    mock_segment.text = "multi frame audio"
    mock_whisper_instance = mock_whisper_model.return_value
    mock_whisper_instance.transcribe.return_value = ([mock_segment], MagicMock())

    listen = Listen()
    result = listen.listen(duration=0.2)

    assert result == "multi frame audio"
    # Should call read multiple times for longer duration
    assert mock_stream.read.call_count > 1


def test_listen_chunk(temp_model_dir, mock_whisper_model, mock_pyaudio):
    """Test listen_chunk method."""
    mock_p = mock_pyaudio.return_value
    mock_stream = MagicMock()
    mock_p.open.return_value = mock_stream

    audio_chunk = np.array([0.1, 0.2], dtype=np.float32).tobytes()
    mock_stream.read.return_value = audio_chunk

    mock_segment = MagicMock()
    mock_segment.text = "chunk audio"
    mock_whisper_instance = mock_whisper_model.return_value
    mock_whisper_instance.transcribe.return_value = ([mock_segment], MagicMock())

    listen = Listen()
    result = listen.listen_chunk(duration=1.0)

    assert result == "chunk audio"


def test_listen_continuous_keyboard_interrupt(
    temp_model_dir, mock_whisper_model, mock_pyaudio, capsys
):
    """Test continuous listening stops on KeyboardInterrupt."""
    mock_p = mock_pyaudio.return_value
    mock_stream = MagicMock()
    mock_p.open.return_value = mock_stream

    audio_chunk = np.array([0.1, 0.2], dtype=np.float32).tobytes()
    mock_stream.read.return_value = audio_chunk

    mock_segment = MagicMock()
    mock_segment.text = "test"

    mock_whisper_instance = mock_whisper_model.return_value

    call_count = 0

    def transcribe_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count > 2:
            raise KeyboardInterrupt()
        return ([mock_segment], MagicMock())

    mock_whisper_instance.transcribe.side_effect = transcribe_side_effect

    listen = Listen()

    # Should not raise, should catch KeyboardInterrupt
    listen.listen_continuous(chunk_duration=0.1)

    captured = capsys.readouterr()
    assert "test" in captured.out


def test_listen_transcribe_multiple_segments(
    temp_model_dir, mock_whisper_model, mock_pyaudio
):
    """Test transcription with multiple segments."""
    mock_p = mock_pyaudio.return_value
    mock_stream = MagicMock()
    mock_p.open.return_value = mock_stream

    audio_chunk = np.array([0.1, 0.2], dtype=np.float32).tobytes()
    mock_stream.read.return_value = audio_chunk

    # Mock multiple segments
    mock_segment1 = MagicMock()
    mock_segment1.text = "hello "
    mock_segment2 = MagicMock()
    mock_segment2.text = "world"

    mock_whisper_instance = mock_whisper_model.return_value
    mock_whisper_instance.transcribe.return_value = (
        [mock_segment1, mock_segment2],
        MagicMock(),
    )

    listen = Listen()
    result = listen.listen(duration=0.1)

    assert result == "hello world"


def test_listen_transcribe_empty_audio(
    temp_model_dir, mock_whisper_model, mock_pyaudio
):
    """Test transcription with silent/empty audio."""
    mock_p = mock_pyaudio.return_value
    mock_stream = MagicMock()
    mock_p.open.return_value = mock_stream

    audio_chunk = np.array([0.0, 0.0], dtype=np.float32).tobytes()
    mock_stream.read.return_value = audio_chunk

    # Empty transcription
    mock_whisper_instance = mock_whisper_model.return_value
    mock_whisper_instance.transcribe.return_value = ([], MagicMock())

    listen = Listen()
    result = listen.listen(duration=0.1)

    assert result == ""


def test_listen_capture_error_handling(
    temp_model_dir, mock_whisper_model, mock_pyaudio
):
    """Test error handling during audio capture."""
    mock_p = mock_pyaudio.return_value
    mock_stream = MagicMock()
    mock_p.open.return_value = mock_stream

    # Simulate audio read error
    mock_stream.read.side_effect = IOError("Audio device error")

    listen = Listen()

    # Should still return result with empty audio
    mock_whisper_instance = mock_whisper_model.return_value
    mock_whisper_instance.transcribe.return_value = ([], MagicMock())

    result = listen.listen(duration=0.1)
    assert isinstance(result, str)


def test_listen_stream_cleanup_on_error(
    temp_model_dir, mock_whisper_model, mock_pyaudio
):
    """Test that stream is properly cleaned up on error."""
    mock_p = mock_pyaudio.return_value
    mock_stream = MagicMock()
    mock_p.open.return_value = mock_stream

    # Simulate IOError during stream read (caught and continues with empty audio)
    mock_stream.read.side_effect = IOError("Audio device error")

    listen = Listen()

    # Setup whisper to not raise on empty audio
    mock_whisper_instance = mock_whisper_model.return_value
    mock_whisper_instance.transcribe.return_value = ([], MagicMock())

    result = listen.listen(duration=0.1)

    # Should return empty string since no audio was captured
    assert result == ""

    # Verify cleanup still happens
    mock_stream.stop_stream.assert_called_once()
    mock_stream.close.assert_called_once()
    mock_p.terminate.assert_called_once()


def test_listen_transcribe_error(temp_model_dir, mock_whisper_model, mock_pyaudio):
    """Test error handling during transcription."""
    mock_p = mock_pyaudio.return_value
    mock_stream = MagicMock()
    mock_p.open.return_value = mock_stream

    audio_chunk = np.array([0.1, 0.2], dtype=np.float32).tobytes()
    mock_stream.read.return_value = audio_chunk

    # Simulate transcription error
    mock_whisper_instance = mock_whisper_model.return_value
    mock_whisper_instance.transcribe.side_effect = Exception("Transcription failed")

    listen = Listen()

    with pytest.raises(Exception, match="Transcription failed"):
        listen.listen(duration=0.1)

    # Verify cleanup still happened
    mock_stream.stop_stream.assert_called_once()
    mock_stream.close.assert_called_once()
    mock_p.terminate.assert_called_once()


def test_listen_audio_format(temp_model_dir, mock_whisper_model, mock_pyaudio):
    """Test that audio is captured with correct format."""
    mock_p = mock_pyaudio.return_value
    mock_stream = MagicMock()
    mock_p.open.return_value = mock_stream

    audio_chunk = np.array([0.1, 0.2], dtype=np.float32).tobytes()
    mock_stream.read.return_value = audio_chunk

    mock_segment = MagicMock()
    mock_segment.text = "test"
    mock_whisper_instance = mock_whisper_model.return_value
    mock_whisper_instance.transcribe.return_value = ([mock_segment], MagicMock())

    listen = Listen()
    listen.listen(duration=0.1)

    # Verify stream was opened with correct parameters
    call_args = mock_p.open.call_args
    assert call_args[1]["format"] == listen.DEFAULT_FORMAT
    assert call_args[1]["channels"] == listen.DEFAULT_CHANNELS
    assert call_args[1]["rate"] == listen.DEFAULT_SAMPLE_RATE
    assert call_args[1]["input"] is True


def test_listen_transcribe_audio_normalization(
    temp_model_dir, mock_whisper_model, mock_pyaudio
):
    """Test that audio data is properly normalized before transcription."""
    mock_p = mock_pyaudio.return_value
    mock_stream = MagicMock()
    mock_p.open.return_value = mock_stream

    audio_chunk = np.array([0.1, 0.2], dtype=np.float32).tobytes()
    mock_stream.read.return_value = audio_chunk

    mock_segment = MagicMock()
    mock_segment.text = "test"
    mock_whisper_instance = mock_whisper_model.return_value
    mock_whisper_instance.transcribe.return_value = ([mock_segment], MagicMock())

    listen = Listen()
    listen.listen(duration=0.1)

    # Get the audio data passed to transcribe
    transcribe_call_args = mock_whisper_instance.transcribe.call_args
    audio_data = transcribe_call_args[0][0]

    # Verify audio is float32 and 1D
    assert audio_data.dtype == np.float32
    assert audio_data.ndim == 1


def test_listen_model_parameters(temp_model_dir, mock_whisper_model):
    """Test that WhisperModel is initialized with correct parameters."""
    Listen("tiny.en")

    call_args = mock_whisper_model.call_args
    assert call_args[0][0] == "tiny.en"
    assert call_args[1]["device"] == "auto"
    assert call_args[1]["compute_type"] == "auto"
    assert str(temp_model_dir) in str(call_args[1]["download_root"])
