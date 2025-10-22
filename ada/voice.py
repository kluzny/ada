import os
import pyaudio

from piper import PiperVoice, SynthesisConfig
from piper.download_voices import download_voice
from pathlib import Path

from ada.logger import build_logger

logger = build_logger(__name__)


class Voice:
    CACHE_DIR = "voices"

    def __init__(self, voice: str):
        """
        Initialize Voice with a voice model identifier.

        Args:
            voice: Voice model identifier (e.g., "en_US-amy-medium")
        """
        self.voice: str = voice
        logger.debug(f"using voice {self.voice}")

        self.__prepare()

        self.voice_config = SynthesisConfig(
            volume=1.0,  # loudness
            length_scale=1.0,  # slowness
            noise_scale=1.0,  # audio variation
            noise_w_scale=1.0,  # speaking variation
            normalize_audio=False,  # use raw audio from voice
        )

        # TODO: eventually need to move use_cude to configuration or auto detection
        # self.piper_voice = PiperVoice.load(self.get_model_path(), use_cuda=True)
        self.piper_voice = PiperVoice.load(self.get_model_path())

    def __prepare(self) -> None:
        """Prepare the voices directory and download voice files if needed."""
        os.makedirs(self.CACHE_DIR, exist_ok=True)

        # Check if voice files already exist
        if not self.__voice_exists():
            logger.info(f"downloading voice {self.voice}...")
            self.__download()
            logger.info(f"voice {self.voice} saved to {self.CACHE_DIR}")
        else:
            logger.info(f"voice {self.voice} exists at {self.CACHE_DIR}")

    def say(self, message: str) -> None:
        """
        Synthesize and play audio message using PyAudio streaming.

        Streams the synthesized audio from Piper directly to the speaker
        without saving to disk. Each audio chunk is played immediately
        as it's generated.

        Args:
            message: The text message to synthesize and play

        Raises:
            Exception: If audio playback fails
        """
        try:
            p = pyaudio.PyAudio()
            stream = None

            try:
                first_chunk = True

                # Stream audio chunks from Piper
                for chunk in self.piper_voice.synthesize(
                    message, syn_config=self.voice_config
                ):
                    # Initialize the audio stream with the first chunk's properties
                    if first_chunk:
                        stream = p.open(
                            format=p.get_format_from_width(chunk.sample_width),
                            channels=chunk.sample_channels,
                            rate=chunk.sample_rate,
                            output=True,
                        )
                        logger.debug(
                            f"opened audio stream: rate={chunk.sample_rate}, "
                            f"channels={chunk.sample_channels}, "
                            f"width={chunk.sample_width}"
                        )
                        first_chunk = False

                    # Write audio data to the stream
                    stream.write(chunk.audio_int16_bytes)

            finally:
                # Clean up resources
                if stream is not None:
                    stream.stop_stream()
                    stream.close()
                p.terminate()

        except Exception as e:
            logger.error(f"failed to play audio: {e}")
            raise

    def __voice_exists(self) -> bool:
        """
        Check if voice model files exist in the cache directory.

        Returns:
            True if both .onnx and .onnx.json files exist, False otherwise
        """
        # Voice files are named like: en_US-amy-medium.onnx and en_US-amy-medium.onnx.json
        model_file = os.path.join(self.CACHE_DIR, f"{self.voice}.onnx")
        config_file = os.path.join(self.CACHE_DIR, f"{self.voice}.onnx.json")

        return os.path.exists(model_file) and os.path.exists(config_file)

    def __download(self) -> None:
        """Download voice model files from Hugging Face."""
        try:
            download_voice(self.voice, Path(self.CACHE_DIR), force_redownload=False)
        except Exception as e:
            logger.error(f"failed to download voice {self.voice}: {e}")
            raise

    def get_model_path(self) -> str:
        """
        Get the path to the downloaded voice model file.

        Returns:
            Path to the .onnx model file
        """
        return os.path.join(self.CACHE_DIR, f"{self.voice}.onnx")

    def get_config_path(self) -> str:
        """
        Get the path to the downloaded voice configuration file.

        Returns:
            Path to the .onnx.json configuration file
        """
        return os.path.join(self.CACHE_DIR, f"{self.voice}.onnx.json")
