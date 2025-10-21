"""
llama-cpp-python backend implementation.

This module provides the LlamaCppBackend class for running local GGUF models.
"""

from typing import Any, List, cast, Optional, Union, Iterator
from llama_cpp import (
    Llama,
    ChatCompletionRequestMessage,
    ChatCompletionRequestResponseFormat,
    ChatCompletionTool,
    ChatCompletionToolChoiceOption,
    CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
)

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

        # Store model information
        self.model_name = model_name
        self.models_list = models

        verbose = config.get("verbose", False)
        n_threads = config.get("threads", 1)
        n_ctx = self.context_window()

        logger.info(f"initializing llama.cpp with model: {self.model.path}")
        logger.info(f"n_ctx: {n_ctx}, n_threads: {n_threads}, verbose: {verbose}")

        self.llm = Llama(
            model_path=self.model.path,
            n_threads=n_threads,
            n_ctx=n_ctx,
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
    ) -> Union[
        CreateChatCompletionResponse, Iterator[CreateChatCompletionStreamResponse]
    ]:
        """
        Generate a chat completion using llama-cpp-python.

        Args:
            messages: List of message dictionaries
            tools: Optional list of tool definitions (dicts will be cast to ChatCompletionTool)
            tool_choice: Tool selection strategy
            response_format: Optional response format (e.g., {"type": "json_object"})
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop: Stop sequences

        Returns:
            The raw response object returned by llama-cpp-python (sync or streaming)
        """
        logger.debug(f"generating completion with {len(messages)} messages")

        return self.llm.create_chat_completion(
            messages=cast(List[ChatCompletionRequestMessage], messages),
            tools=cast(Optional[List[ChatCompletionTool]], tools),
            tool_choice=cast(Optional[ChatCompletionToolChoiceOption], tool_choice),
            response_format=cast(
                Optional[ChatCompletionRequestResponseFormat], response_format
            ),
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
        )

    # def chat_completion(
    #     self,
    #     messages: list[dict],
    #     tools: list[dict] | None = None,
    #     tool_choice: str | dict = "auto",
    #     response_format: dict | None = None,
    #     temperature: float = 0.7,
    #     max_tokens: int | None = None,
    #     stop: list[str] | None = None,
    # ) -> dict:
    #     """
    #     Generate a chat completion using llama-cpp-python.

    #     Args:
    #         messages: List of message dictionaries
    #         tools: Optional list of tool definitions
    #         tool_choice: Tool selection strategy
    #         response_format: Optional response format (e.g., {"type": "json_object"})
    #         temperature: Sampling temperature
    #         max_tokens: Maximum tokens to generate
    #         stop: Stop sequences

    #     Returns:
    #         Response dictionary in OpenAI-compatible format
    #     """
    #     logger.debug(f"generating completion with {len(messages)} messages")

    #     return self.llm.create_chat_completion(
    #         messages=cast(List[ChatCompletionRequestMessage], messages),
    #         tools=tools,
    #         tool_choice=tool_choice,
    #         response_format=response_format,
    #         temperature=temperature,
    #         max_tokens=max_tokens,
    #         stop=stop,
    #     )

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

    def context_window(self) -> int:
        """
        Get the context window size from the GGUF model metadata on an llm instance.
        Falls back to 2048

        Returns:
            The context window size in tokens
        """
        try:
            return self.__get_maximum_context_from_llm_instance(self.model.path)
        except Exception as e:
            logger.warning(
                f"Failed to get context window size: {e}, defaulting to 2048"
            )
            return 2048

    def __get_maximum_context_from_llm_instance(self, path: str) -> int:
        # only exists long enough to grab the metadata, magic params to keep it quiet
        info_llm = Llama(model_path=path, verbose=False, n_ctx=0)
        if hasattr(info_llm, "metadata") and isinstance(info_llm.metadata, dict):
            # Check for llama.context_length in metadata
            context_length = info_llm.metadata.get("llama.context_length")
            if context_length is not None:
                # Metadata values are strings, convert to int
                return int(context_length)
        raise ValueError("unable to inspect `llama.context_length` from LLM metadata")

    def __str__(self) -> str:
        return f"LlamaCppBackend(model={self.model.name if hasattr(self, 'model') else 'unknown'})"
