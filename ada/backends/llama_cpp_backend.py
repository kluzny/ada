"""
llama-cpp-python backend implementation.

This module provides the LlamaCppBackend class for running local GGUF models.
"""

from typing import Any
from llama_cpp import Llama

from .base import Base
from ada.model import Model
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
                - model: Name of the model to use
                - models: Array of model definitions with name, url, tokens
                - threads: Number of threads to use (default: 4)
                - verbose: Whether to enable verbose output (default: False)
        """
        super().__init__(config)

        # Get the model name
        model_name = config.get("model")
        if not model_name:
            raise ValueError("'model' is required in llama-cpp backend configuration")

        # Find the model definition in the models array
        models = config.get("models", [])
        model_def = None
        for m in models:
            if m.get("name") == model_name:
                model_def = m
                break

        if not model_def:
            raise ValueError(f"Model '{model_name}' not found in models array")

        # Extract configuration
        model_url = model_def.get("url")
        if not model_url:
            raise ValueError(f"Model '{model_name}' is missing 'url'")

        # Use Model class to handle downloading and caching
        self.model = Model(model_url)
        model_path = self.model.path

        # Store model information
        self.model_name = model_name
        self.models_list = models

        n_ctx = model_def.get("tokens", 2048)
        n_threads = config.get("threads", 4)
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

    def current_model(self) -> str:
        """
        Get the name of the currently active model.

        Returns:
            The name of the current model
        """
        return self.model_name

    def available_models(self) -> list[str]:
        """
        Get a list of available model names from the configuration.

        Returns:
            List of model names that can be used
        """
        return [model.get("name") for model in self.models_list if model.get("name")]

    def __str__(self) -> str:
        return f"LlamaCppBackend(model={self.model.name if hasattr(self, 'model') else 'unknown'})"
