"""
Base class for LLM backends.

This module provides the Base class that all backend implementations should inherit from.
"""

from abc import ABC, abstractmethod
from typing import Any


class Base(ABC):
    """
    Base class for all LLM backends.

    This class provides a common interface for different LLM backend implementations
    (e.g., llama-cpp-python, Ollama). Subclasses should override the chat_completion
    method to implement their specific backend logic.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize the backend.

        Args:
            config: A dictionary containing backend-specific configuration
        """
        self.config = config

    @abstractmethod
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
        Generate a chat completion response.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            tools: Optional list of tool definitions for function calling
            tool_choice: How the model should choose tools ("auto", "none", or specific tool)
            response_format: Optional response format specification (e.g., {"type": "json_object"})
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            stop: List of stop sequences

        Returns:
            A dictionary containing the response in OpenAI-compatible format with structure:
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": str | None,
                            "tool_calls": [...] (optional)
                        }
                    }
                ],
                "usage": {
                    "total_tokens": int
                }
            }
        """
        raise NotImplementedError("Subclasses must implement chat_completion method")

    @abstractmethod
    def current_model(self) -> str:
        """
        Get the name of the currently active model.

        Returns:
            The name of the current model
        """
        raise NotImplementedError("Subclasses must implement current_model method")

    @abstractmethod
    def available_models(self) -> list[str]:
        """
        Get a list of available model names.

        Returns:
            List of model names that can be used with this backend
        """
        raise NotImplementedError("Subclasses must implement available_models method")

    @abstractmethod
    def context_window(self) -> int:
        """
        Get the context window size (maximum tokens) for the current model.

        Returns:
            The context window size in tokens. Defaults to 2048 if unable to determine.
        """
        raise NotImplementedError("Subclasses must implement context_window method")

    def __str__(self) -> str:
        return f"{self.__class__.__name__}"
