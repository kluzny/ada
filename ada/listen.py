import os
import numpy as np
import pyaudio
from faster_whisper import WhisperModel

from ada.logger import build_logger
from ada.constants import ARTIFACT_DIR

logger = build_logger(__name__)


class Listen:
    """
    Capture streaming audio from microphone and transcribe to text using faster-whisper.

    Uses PyAudio to capture audio in real-time and the faster-whisper medium.en model
    for efficient speech-to-text transcription. Supports both continuous and chunked
    transcription modes.
    """

    CACHE_DIR = ARTIFACT_DIR / "whisper"
    DEFAULT_SAMPLE_RATE = 16000
    DEFAULT_CHUNK_SIZE = 1024
    DEFAULT_FORMAT = pyaudio.paFloat32
    DEFAULT_CHANNELS = 1

    def __init__(self, model: str = "medium.en"):
        """
        Initialize Listen with a faster-whisper model.

        Args:
            model: Model identifier (default: "medium.en")
                   Options: "tiny", "tiny.en", "base", "base.en", "small", "small.en",
                           "medium", "medium.en", "large", "large-v1", "large-v2"
        """
        self.model_name = model
        logger.debug(f"using whisper model {self.model_name}")

        self.__prepare()

        # Load the model - device auto-detects GPU/CPU
        # compute_type auto-selects based on available hardware
        self.model = WhisperModel(
            self.model_name,
            device="auto",
            compute_type="auto",
            download_root=str(self.CACHE_DIR),
        )
        logger.info(f"loaded whisper model {self.model_name}")

    def listen(self, duration: float = 10.0) -> str:
        """
        Capture audio from microphone for specified duration and transcribe to text.

        Args:
            duration: Recording duration in seconds (default: 10.0)

        Returns:
            Transcribed text from the audio

        Raises:
            Exception: If audio capture or transcription fails
        """
        try:
            audio_data = self.__capture_audio(duration)
            return self.__transcribe(audio_data)
        except Exception as e:
            logger.error(f"failed to listen and transcribe: {e}")
            raise

    def listen_chunk(self, duration: float = 3.0) -> str:
        """
        Capture a single audio chunk and return transcribed text.

        Useful for continuous transcription where you want to process
        audio in smaller intervals.

        Args:
            duration: Chunk duration in seconds (default: 3.0)

        Returns:
            Transcribed text from the audio chunk
        """
        try:
            audio_data = self.__capture_audio(duration)
            return self.__transcribe(audio_data)
        except Exception as e:
            logger.error(f"failed to listen to chunk: {e}")
            raise

    def listen_continuous(self, chunk_duration: float = 3.0) -> None:
        """
        Continuously capture and transcribe audio chunks.

        Runs indefinitely, transcribing and printing audio chunks as they're captured.
        Press Ctrl+C to stop.

        Args:
            chunk_duration: Duration of each chunk in seconds (default: 3.0)
        """
        logger.info("starting continuous listening (press Ctrl+C to stop)")
        try:
            while True:
                text = self.listen_chunk(chunk_duration)
                if text.strip():
                    print(f"[whisper] {text}")
        except KeyboardInterrupt:
            logger.info("continuous listening stopped")
        except Exception as e:
            logger.error(f"failed during continuous listening: {e}")
            raise

    def __prepare(self) -> None:
        """Prepare the models directory for whisper model caching."""
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        logger.debug(f"whisper models cache directory: {self.CACHE_DIR}")

    def __capture_audio(self, duration: float) -> np.ndarray:
        """
        Capture audio from the microphone.

        Args:
            duration: Recording duration in seconds

        Returns:
            Audio data as numpy array (float32, mono)

        Raises:
            Exception: If audio capture fails
        """
        p = pyaudio.PyAudio()
        stream = None

        try:
            stream = p.open(
                format=self.DEFAULT_FORMAT,
                channels=self.DEFAULT_CHANNELS,
                rate=self.DEFAULT_SAMPLE_RATE,
                input=True,
                frames_per_buffer=self.DEFAULT_CHUNK_SIZE,
            )

            logger.debug(
                f"opened audio stream: rate={self.DEFAULT_SAMPLE_RATE}, "
                f"channels={self.DEFAULT_CHANNELS}, "
                f"format={self.DEFAULT_FORMAT}"
            )

            frames = []
            num_frames = int(
                self.DEFAULT_SAMPLE_RATE / self.DEFAULT_CHUNK_SIZE * duration
            )

            logger.debug(f"capturing {duration}s of audio ({num_frames} frames)")

            for i in range(num_frames):
                try:
                    data = stream.read(
                        self.DEFAULT_CHUNK_SIZE, exception_on_overflow=False
                    )
                    frames.append(np.frombuffer(data, dtype=np.float32))
                except IOError as e:
                    logger.warning(f"audio frame {i} read error: {e}, continuing")
                    continue

            # Concatenate all frames into single array
            audio_data = (
                np.concatenate(frames) if frames else np.array([], dtype=np.float32)
            )
            logger.debug(f"captured {len(audio_data)} samples")

            return audio_data

        finally:
            if stream is not None:
                stream.stop_stream()
                stream.close()
            p.terminate()

    def __transcribe(self, audio_data: np.ndarray) -> str:
        """
        Transcribe audio data using faster-whisper.

        Args:
            audio_data: Audio data as numpy array

        Returns:
            Transcribed text

        Raises:
            Exception: If transcription fails
        """
        try:
            logger.debug(f"transcribing {len(audio_data)} samples")

            # Ensure audio is mono and float32
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)
            audio_data = audio_data.astype(np.float32)

            # Transcribe using faster-whisper
            segments, info = self.model.transcribe(
                audio_data,
                language="en",
                beam_size=5,
            )

            # Collect all segments into a single text
            text = "".join([segment.text for segment in segments]).strip()

            logger.debug(f"transcription complete: {text[:100]}")

            return text

        except Exception as e:
            logger.error(f"failed to transcribe audio: {e}")
            raise


if __name__ == "__main__":
    """Test the Listen class with continuous listening."""
    import sys

    print("Starting continuous listening...")
    print("Speak into your microphone. Press Ctrl+C to stop.\n")

    try:
        listen = Listen(model="medium.en")
        listen.listen_continuous(chunk_duration=3.0)
    except KeyboardInterrupt:
        print("\n\nListening stopped.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError during listening: {e}")
        sys.exit(1)
