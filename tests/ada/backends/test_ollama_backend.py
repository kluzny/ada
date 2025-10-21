from typing import cast
from ollama._types import ChatResponse, Message

import pytest
from unittest.mock import Mock, patch
from ada.backends.ollama_backend import OllamaBackend


@pytest.fixture
def sample_config():
    """Sample configuration for ollama backend."""
    return {
        "model": "llama2",
        "url": "http://localhost:11434",
    }


@pytest.fixture
def mock_ollama_client():
    """Mock ollama Client class."""
    with patch("ada.backends.ollama_backend.ollama.Client") as mock:
        mock_instance = Mock()
        mock.return_value = mock_instance
        yield mock_instance


def test_ollama_backend_initialization(sample_config, mock_ollama_client):
    """Test OllamaBackend initialization."""
    backend = OllamaBackend(sample_config)

    assert backend.model_name == "llama2"
    assert backend.url == "http://localhost:11434"
    assert backend.client == mock_ollama_client


def test_ollama_backend_default_url(mock_ollama_client):
    """Test OllamaBackend with default URL."""
    config = {"model": "llama2"}

    backend = OllamaBackend(config)

    assert backend.url == "http://localhost:11434"


def test_ollama_backend_missing_model(mock_ollama_client):
    """Test OllamaBackend with missing model key."""
    config = {"url": "http://localhost:11434"}

    with pytest.raises(ValueError, match="'model' is required"):
        OllamaBackend(config)


def test_current_model(sample_config, mock_ollama_client):
    """Test current_model method."""
    backend = OllamaBackend(sample_config)

    assert backend.current_model() == "llama2"


def test_available_models_success(sample_config, mock_ollama_client):
    """Test available_models method with successful response."""
    mock_ollama_client.list.return_value = {
        "models": [
            {"name": "llama2"},
            {"name": "mistral"},
            {"name": "codellama"},
        ]
    }

    backend = OllamaBackend(sample_config)
    models = backend.available_models()

    assert len(models) == 3
    assert "llama2" in models
    assert "mistral" in models
    assert "codellama" in models
    mock_ollama_client.list.assert_called_once()


def test_available_models_empty_response(sample_config, mock_ollama_client):
    """Test available_models with empty response."""
    mock_ollama_client.list.return_value = {"models": []}

    backend = OllamaBackend(sample_config)
    models = backend.available_models()

    assert len(models) == 0


def test_available_models_failure(sample_config, mock_ollama_client):
    """Test available_models when API call fails."""
    mock_ollama_client.list.side_effect = Exception("Connection error")

    backend = OllamaBackend(sample_config)
    models = backend.available_models()

    # Should return at least the configured model as fallback
    assert len(models) == 1
    assert "llama2" in models


def test_chat_completion_without_tools(sample_config, mock_ollama_client):
    """Test chat_completion method without tools."""
    backend = OllamaBackend(sample_config)

    ollama_response = cast(
        ChatResponse,
        {
            "message": {"role": "assistant", "content": "Hello!"},
            "done": True,
            "prompt_eval_count": 10,
            "eval_count": 5,
        },
    )

    mock_ollama_client.chat.return_value = ollama_response

    messages = [{"role": "user", "content": "Hello"}]
    response = backend.chat_completion(messages)

    assert "choices" in response
    assert response["choices"][0]["message"]["content"] == "Hello!"
    assert response["usage"]["total_tokens"] == 15

    mock_ollama_client.chat.assert_called_once()
    call_args = mock_ollama_client.chat.call_args
    assert call_args.kwargs["model"] == "llama2"
    assert call_args.kwargs["messages"] == messages


def test_chat_completion_with_tools(sample_config, mock_ollama_client):
    """Test chat_completion method with tools."""
    backend = OllamaBackend(sample_config)

    tools = [{"type": "function", "function": {"name": "test_tool"}}]
    ollama_response = cast(
        ChatResponse,
        {
            "message": {"role": "assistant", "content": "Using tool"},
            "done": True,
            "prompt_eval_count": 10,
            "eval_count": 5,
        },
    )
    mock_ollama_client.chat.return_value = ollama_response

    messages = [{"role": "user", "content": "Test"}]
    response = backend.chat_completion(messages, tools=tools)

    assert "choices" in response

    call_args = mock_ollama_client.chat.call_args
    assert call_args.kwargs["tools"] == tools


def test_chat_completion_with_json_format(sample_config, mock_ollama_client):
    """Test chat_completion with JSON response format."""
    backend = OllamaBackend(sample_config)

    ollama_response = cast(
        ChatResponse,
        {
            "message": {"role": "assistant", "content": '{"key": "value"}'},
            "done": True,
            "prompt_eval_count": 10,
            "eval_count": 5,
        },
    )
    mock_ollama_client.chat.return_value = ollama_response

    messages = [{"role": "user", "content": "Test"}]
    response = backend.chat_completion(
        messages, response_format={"type": "json_object"}
    )

    assert "choices" in response

    call_args = mock_ollama_client.chat.call_args
    assert call_args.kwargs["format"] == "json"


def test_chat_completion_error(sample_config, mock_ollama_client):
    """Test chat_completion when API call fails."""
    backend = OllamaBackend(sample_config)
    mock_ollama_client.chat.side_effect = Exception("API error")

    messages = [{"role": "user", "content": "Test"}]

    with pytest.raises(Exception, match="API error"):
        backend.chat_completion(messages)


def test_convert_response_with_tool_calls(sample_config, mock_ollama_client):
    """Test response conversion with tool calls."""
    backend = OllamaBackend(sample_config)

    tool_call = Message.ToolCall(
        function=Message.ToolCall.Function(name="test", arguments={})
    )

    ollama_response = cast(
        ChatResponse,
        {
            "message": {
                "role": "assistant",
                "content": "Using tool",
                "tool_calls": [tool_call],
            },
            "done": True,
            "prompt_eval_count": 10,
            "eval_count": 5,
        },
    )
    openai_response = backend._convert_response(ollama_response)

    assert "tool_calls" in openai_response["choices"][0]["message"]
    assert len(openai_response["choices"][0]["message"]["tool_calls"]) == 1
    # Verify the tool call was converted to OpenAI format
    assert (
        openai_response["choices"][0]["message"]["tool_calls"][0]["type"] == "function"
    )
    assert "function" in openai_response["choices"][0]["message"]["tool_calls"][0]
    assert (
        openai_response["choices"][0]["message"]["tool_calls"][0]["function"]["name"]
        == "test"
    )


def test_str_representation(sample_config, mock_ollama_client):
    """Test string representation."""
    backend = OllamaBackend(sample_config)

    assert "OllamaBackend" in str(backend)
    assert "llama2" in str(backend)
    assert "http://localhost:11434" in str(backend)


def test_chat_completion_with_gpt_oss():
    """Test chat_completion with gpt-oss model without mocks."""
    config = {
        "model": "gpt-oss:20b",
        "url": "http://localhost:11434",
    }

    backend = OllamaBackend(config)

    messages = [{"role": "user", "content": "Hello"}]

    # Call chat_completion and verify it doesn't raise an error
    response = backend.chat_completion(messages, max_tokens=50)

    # Verify response is a dict (JSON object)
    assert isinstance(response, dict)

    # Verify response has expected OpenAI-compatible structure
    assert "choices" in response
    assert isinstance(response["choices"], list)
    assert len(response["choices"]) > 0
    assert "message" in response["choices"][0]
    assert "content" in response["choices"][0]["message"]

    # Verify usage information is present
    assert "usage" in response
    assert "total_tokens" in response["usage"]


def test_context_window_with_gpt_oss():
    """Test context_window retrieves value from Ollama model metadata."""
    config = {
        "model": "gpt-oss:20b",
        "url": "http://localhost:11434",
    }

    backend = OllamaBackend(config)

    # Get context window from the backend
    context_size = backend.context_window()

    # Should return an integer
    assert isinstance(context_size, int)

    # Should be greater than 0
    assert context_size > 0

    # gpt-oss should have a reasonable context window (at least 2048)
    assert context_size >= 2048


def test_context_window_from_model_info_num_ctx(sample_config, mock_ollama_client):
    """Test context_window extracts num_ctx from model_info."""
    backend = OllamaBackend(sample_config)

    # Mock client.show to return model_info with num_ctx
    mock_ollama_client.show.return_value = {"model_info": {"num_ctx": 4096}}

    context_size = backend.context_window()

    assert context_size == 4096
    mock_ollama_client.show.assert_called_once_with("llama2")


def test_context_window_from_model_info_context_length(
    sample_config, mock_ollama_client
):
    """Test context_window extracts context_length from model_info."""
    backend = OllamaBackend(sample_config)

    # Mock client.show to return model_info with context_length
    mock_ollama_client.show.return_value = {"model_info": {"context_length": 8192}}

    context_size = backend.context_window()

    assert context_size == 8192
    mock_ollama_client.show.assert_called_once_with("llama2")


def test_context_window_from_parameters(sample_config, mock_ollama_client):
    """Test context_window extracts num_ctx from parameters."""
    backend = OllamaBackend(sample_config)

    # Mock client.show to return parameters with num_ctx
    mock_ollama_client.show.return_value = {"parameters": {"num_ctx": "16384"}}

    context_size = backend.context_window()

    assert context_size == 16384
    mock_ollama_client.show.assert_called_once_with("llama2")


def test_context_window_fallback_when_show_fails(sample_config, mock_ollama_client):
    """Test context_window returns 2048 when client.show fails."""
    backend = OllamaBackend(sample_config)

    # Mock client.show to raise exception
    mock_ollama_client.show.side_effect = Exception("Connection error")

    context_size = backend.context_window()

    assert context_size == 2048


def test_context_window_with_no_num_ctx(sample_config, mock_ollama_client):
    """Test context_window returns 2048 when num_ctx not found."""
    backend = OllamaBackend(sample_config)

    # Mock client.show to return response without num_ctx
    mock_ollama_client.show.return_value = {
        "model_info": {"some_other_field": "value"},
        "parameters": {"temperature": 0.7},
    }

    context_size = backend.context_window()

    assert context_size == 2048


def test_context_window_with_empty_model_info(sample_config, mock_ollama_client):
    """Test context_window handles empty model_info."""
    backend = OllamaBackend(sample_config)

    # Mock client.show to return empty model_info
    mock_ollama_client.show.return_value = {"model_info": {}}

    context_size = backend.context_window()

    assert context_size == 2048


def test_context_window_with_non_dict_model_info(sample_config, mock_ollama_client):
    """Test context_window handles non-dict model_info."""
    backend = OllamaBackend(sample_config)

    # Mock client.show to return non-dict model_info
    mock_ollama_client.show.return_value = {"model_info": "not a dict"}

    context_size = backend.context_window()

    assert context_size == 2048


def test_convert_response_with_missing_message(sample_config, mock_ollama_client):
    """Test _convert_response handles missing message gracefully."""
    backend = OllamaBackend(sample_config)

    # Ollama response without message field
    ollama_response = cast(
        ChatResponse,
        {
            "done": True,
            "prompt_eval_count": 10,
            "eval_count": 5,
        },
    )

    openai_response = backend._convert_response(ollama_response)

    # Should have valid structure with empty/None content
    assert "choices" in openai_response
    assert len(openai_response["choices"]) == 1
    assert "message" in openai_response["choices"][0]
    assert openai_response["choices"][0]["message"]["role"] == "assistant"
    assert openai_response["choices"][0]["message"]["content"] is None
    assert "usage" in openai_response
    assert openai_response["usage"]["total_tokens"] == 15


def test_convert_response_with_zero_tokens(sample_config, mock_ollama_client):
    """Test _convert_response handles zero token counts."""
    backend = OllamaBackend(sample_config)

    # Ollama response with zero tokens
    ollama_response = cast(
        ChatResponse,
        {
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "prompt_eval_count": 0,
            "eval_count": 0,
        },
    )

    openai_response = backend._convert_response(ollama_response)

    # Should calculate usage correctly
    assert openai_response["usage"]["prompt_tokens"] == 0
    assert openai_response["usage"]["completion_tokens"] == 0
    assert openai_response["usage"]["total_tokens"] == 0


def test_convert_response_with_missing_token_counts(sample_config, mock_ollama_client):
    """Test _convert_response handles missing token counts."""
    backend = OllamaBackend(sample_config)

    # Ollama response without token count fields
    ollama_response = cast(
        ChatResponse,
        {
            "message": {"role": "assistant", "content": "test response"},
            "done": True,
        },
    )

    openai_response = backend._convert_response(ollama_response)

    # Should default to 0 for missing counts
    assert openai_response["usage"]["prompt_tokens"] == 0
    assert openai_response["usage"]["completion_tokens"] == 0
    assert openai_response["usage"]["total_tokens"] == 0


def test_convert_response_preserves_tool_calls(sample_config, mock_ollama_client):
    """Test _convert_response converts tool_calls to OpenAI format."""
    backend = OllamaBackend(sample_config)

    # Ollama response with tool_calls (Message.ToolCall objects)
    tool_calls = [
        Message.ToolCall(
            function=Message.ToolCall.Function(
                name="get_weather", arguments={"location": "SF"}
            )
        )
    ]

    ollama_response = cast(
        ChatResponse,
        {
            "message": {
                "role": "assistant",
                "content": "Let me check the weather",
                "tool_calls": tool_calls,
            },
            "done": True,
            "prompt_eval_count": 20,
            "eval_count": 10,
        },
    )

    openai_response = backend._convert_response(ollama_response)

    # Should convert tool_calls to OpenAI format
    assert "tool_calls" in openai_response["choices"][0]["message"]
    assert len(openai_response["choices"][0]["message"]["tool_calls"]) == 1
    tool_call = openai_response["choices"][0]["message"]["tool_calls"][0]
    assert tool_call["type"] == "function"
    assert tool_call["function"]["name"] == "get_weather"
    # Arguments should be JSON string in OpenAI format
    assert isinstance(tool_call["function"]["arguments"], str)


def test_convert_response_with_empty_message_dict(sample_config, mock_ollama_client):
    """Test _convert_response with empty message dictionary."""
    backend = OllamaBackend(sample_config)

    ollama_response = cast(
        ChatResponse,
        {
            "message": {},
            "done": True,
        },
    )

    openai_response = backend._convert_response(ollama_response)

    # Should handle gracefully
    assert "choices" in openai_response
    assert openai_response["choices"][0]["message"]["role"] == "assistant"
    assert openai_response["choices"][0]["message"]["content"] is None
