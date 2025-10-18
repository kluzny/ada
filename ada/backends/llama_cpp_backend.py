"""
llama-cpp-python backend implementation.

This module provides the LlamaCppBackend class for running local GGUF models.
"""

from typing import Any
from llama_cpp import Llama

from .base import Base
from ada.logger import build_logger

logger = build_logger(__name__)


class LlamaCppBackend(Base):
    """
    Backend implementation using llama-cpp-python for local GGUF models.

    This backend downloads and runs GGUF models locally using the llama.cpp library.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize the llama-cpp-python backend.

        Args:
            config: Configuration dictionary with keys:
                - model_path: Path to the GGUF model file
                - n_ctx: Context window size (default: 2048)
                - n_threads: Number of threads to use (default: 4)
                - verbose: Whether to enable verbose output (default: False)
        """
        super().__init__(config)

        model_path = config.get("model_path")
        if not model_path:
            raise ValueError("model_path is required for LlamaCppBackend")

        n_ctx = config.get("n_ctx", 2048)
        n_threads = config.get("n_threads", 4)
        verbose = config.get("verbose", False)

        logger.info(f"initializing llama.cpp with model: {model_path}")
        logger.info(f"context window: {n_ctx}, threads: {n_threads}")

        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=verbose,
        )

    def chat_completion(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict = "auto",
        response_format: dict | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> dict:
        """
        Generate a chat completion using llama-cpp-python.

        Args:
            messages: List of message dictionaries
            tools: Optional list of tool definitions
            tool_choice: Tool selection strategy
            response_format: Optional response format (e.g., {"type": "json_object"})
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop: Stop sequences

        Returns:
            Response dictionary in OpenAI-compatible format
        """
        logger.debug(f"generating completion with {len(messages)} messages")

        return self.llm.create_chat_completion(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
        )

    def __str__(self) -> str:
        return f"LlamaCppBackend(model={self.config.get('model_path', 'unknown')})"
