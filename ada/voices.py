import os

from piper.download_voices import download_voice
from pathlib import Path

from ada.logger import build_logger

logger = build_logger(__name__)


class Voices:
    CACHE_DIR = "voices"

    def __init__(self, voice: str):
        """
        Initialize Voices with a voice model identifier.

        Args:
            voice: Voice model identifier (e.g., "en_US-amy-medium")
        """
        self.voice: str = voice
        logger.debug(f"using voice {self.voice}")

        self.__prepare()

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
