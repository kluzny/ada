"""
Ollama backend implementation.

This module provides the OllamaBackend class for using models via Ollama.
"""

from typing import Any
import ollama

from .base import Base
from ada.logger import build_logger

logger = build_logger(__name__)


class OllamaBackend(Base):
    """
    Backend implementation using Ollama for local model serving.

    This backend connects to a running Ollama instance to generate completions.
    Ollama must be running separately (e.g., `ollama serve`).
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize the Ollama backend.

        Args:
            config: Configuration dictionary with keys:
                - model: Name of the Ollama model (e.g., "llama2", "mistral")
                - url: Ollama server URL (default: http://localhost:11434)
                - tokens: Context window size (for warnings)
        """
        super().__init__(config)

        self.model_name = config.get("model")
        if not self.model_name:
            raise ValueError("'model' is required in ollama backend configuration")

        # Get the Ollama server URL
        self.url = config.get("url", "http://localhost:11434")

        self.client = ollama.Client(host=self.url)
        logger.info(
            f"initializing Ollama backend with model: {self.model_name}, url: {self.url}"
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
        Generate a chat completion using Ollama.

        Args:
            messages: List of message dictionaries
            tools: Optional list of tool definitions
            tool_choice: Tool selection strategy (note: Ollama support may vary)
            response_format: Optional response format
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop: Stop sequences

        Returns:
            Response dictionary in OpenAI-compatible format
        """
        logger.debug(f"generating completion with {len(messages)} messages")

        # Build options dict for Ollama
        options = {
            "temperature": temperature,
        }

        if stop:
            options["stop"] = stop

        # Ollama uses "num_predict" instead of "max_tokens"
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        # Call Ollama
        try:
            # Check if tools are provided
            if tools and len(tools) > 0:
                response = self.client.chat(
                    model=self.model_name,
                    messages=messages,
                    tools=tools,
                    options=options,
                    format="json"
                    if response_format and response_format.get("type") == "json_object"
                    else None,
                )
            else:
                response = self.client.chat(
                    model=self.model_name,
                    messages=messages,
                    options=options,
                    format="json"
                    if response_format and response_format.get("type") == "json_object"
                    else None,
                )

            # Convert Ollama response to OpenAI-compatible format
            return self._convert_response(response)

        except Exception as e:
            logger.error(f"Ollama error: {e}")
            raise

    def _convert_response(self, ollama_response: dict) -> dict:
        """
        Convert Ollama response format to OpenAI-compatible format.

        Ollama response structure:
        {
            "message": {
                "role": "assistant",
                "content": "...",
                "tool_calls": [...] (optional)
            },
            "done": true,
            "total_duration": ...,
            "prompt_eval_count": ...,
            "eval_count": ...
        }

        Args:
            ollama_response: The response from Ollama

        Returns:
            OpenAI-compatible response dictionary
        """
        message = ollama_response.get("message", {})

        # Calculate total tokens (approximation)
        prompt_tokens = ollama_response.get("prompt_eval_count", 0)
        completion_tokens = ollama_response.get("eval_count", 0)
        total_tokens = prompt_tokens + completion_tokens

        # Build OpenAI-compatible response
        openai_response = {
            "choices": [
                {
                    "message": {
                        "role": message.get("role", "assistant"),
                        "content": message.get("content"),
                    }
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        }

        # Include tool_calls if present
        if "tool_calls" in message:
            openai_response["choices"][0]["message"]["tool_calls"] = message[
                "tool_calls"
            ]

        return openai_response

    def current_model(self) -> str:
        """
        Get the name of the currently active model.

        Returns:
            The name of the current model
        """
        return self.model_name

    def available_models(self) -> list[str]:
        """
        Get a list of available models from the Ollama server.

        Returns:
            List of model names available on the Ollama server
        """
        try:
            models = self.client.list()
            return [model["name"] for model in models.get("models", [])]
        except Exception as e:
            logger.warning(f"Failed to list models from Ollama server: {e}")
            return [self.model_name]  # Return at least the configured model

    def context_window(self) -> int:
        """
        Get the context window size from the model metadata.

        Attempts to retrieve num_ctx from the model using client.show.
        Falls back to 2048 if unable to determine.

        Returns:
            The context window size in tokens
        """
        try:
            model_info = self.client.show(self.model_name)
            # Try to get num_ctx from model_info
            if "model_info" in model_info:
                model_data = model_info["model_info"]
                # Check various possible locations for context size
                if isinstance(model_data, dict):
                    # Some models store it directly
                    if "num_ctx" in model_data:
                        return model_data["num_ctx"]
                    # Some store it in a context_length field
                    if "context_length" in model_data:
                        return model_data["context_length"]

            # Try to get from parameters
            if "parameters" in model_info:
                params = model_info["parameters"]
                if "num_ctx" in params:
                    return int(params["num_ctx"])

            logger.debug(f"Could not find num_ctx in model metadata, defaulting to 2048")
            return 2048
        except Exception as e:
            logger.warning(f"Failed to get context window from Ollama: {e}, defaulting to 2048")
            return 2048

    def __str__(self) -> str:
        return f"OllamaBackend(model={self.model_name}, url={self.url})"
