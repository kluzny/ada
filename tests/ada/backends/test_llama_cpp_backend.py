import pytest
from unittest.mock import Mock, patch
from ada.backends.llama_cpp_backend import LlamaCppBackend


@pytest.fixture
def sample_config():
    """Sample configuration for llama-cpp backend."""
    return {
        "model": "test-model",
        "threads": 2,
        "verbose": False,
        "models": [
            {
                "name": "test-model",
                "url": "https://example.com/test-model.gguf",
                "tokens": 1024,
            },
            {
                "name": "another-model",
                "url": "https://example.com/another-model.gguf",
                "tokens": 2048,
            },
        ],
    }


@pytest.fixture
def mock_model():
    """Mock Model class."""
    with patch("ada.backends.llama_cpp_backend.Model") as mock:
        mock_instance = Mock()
        mock_instance.path = "/path/to/model.gguf"
        mock_instance.name = "test-model.gguf"
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def mock_llama():
    """Mock Llama class."""
    with patch("ada.backends.llama_cpp_backend.Llama") as mock:
        mock_instance = Mock()
        mock.return_value = mock_instance
        yield mock


def test_llama_cpp_backend_initialization(sample_config, mock_model, mock_llama):
    """Test LlamaCppBackend initialization."""
    backend = LlamaCppBackend(sample_config)

    assert backend.model_name == "test-model"
    assert len(backend.models_list) == 2
    mock_model.assert_called_once_with("https://example.com/test-model.gguf")
    mock_llama.assert_called_once()


def test_llama_cpp_backend_missing_model(mock_model, mock_llama):
    """Test LlamaCppBackend with missing model key."""
    config = {"models": []}

    with pytest.raises(ValueError, match="'model' is required"):
        LlamaCppBackend(config)


def test_llama_cpp_backend_model_not_found(mock_model, mock_llama):
    """Test LlamaCppBackend with model not in models array."""
    config = {
        "model": "nonexistent-model",
        "models": [{"name": "other-model", "url": "https://example.com/other.gguf"}],
    }

    with pytest.raises(ValueError, match="Model 'nonexistent-model' not found"):
        LlamaCppBackend(config)


def test_llama_cpp_backend_model_missing_url(mock_model, mock_llama):
    """Test LlamaCppBackend with model missing url."""
    config = {"model": "test-model", "models": [{"name": "test-model"}]}

    with pytest.raises(ValueError, match="is missing 'url'"):
        LlamaCppBackend(config)


def test_current_model(sample_config, mock_model, mock_llama):
    """Test current_model method."""
    backend = LlamaCppBackend(sample_config)

    assert backend.current_model() == "test-model"


def test_available_models(sample_config, mock_model, mock_llama):
    """Test available_models method."""
    backend = LlamaCppBackend(sample_config)

    models = backend.available_models()
    assert len(models) == 2
    assert "test-model" in models
    assert "another-model" in models


def test_available_models_filters_invalid(mock_model, mock_llama):
    """Test available_models filters out entries without names."""
    config = {
        "model": "test-model",
        "models": [
            {"name": "test-model", "url": "https://example.com/test.gguf"},
            {"url": "https://example.com/noname.gguf"},  # Missing name
            {"name": "valid-model", "url": "https://example.com/valid.gguf"},
        ],
    }

    backend = LlamaCppBackend(config)
    models = backend.available_models()

    assert len(models) == 2
    assert "test-model" in models
    assert "valid-model" in models


def test_chat_completion(sample_config, mock_model, mock_llama):
    """Test chat_completion method."""
    backend = LlamaCppBackend(sample_config)
    mock_llm = mock_llama.return_value

    expected_response = {"choices": [{"message": {"content": "test"}}]}
    mock_llm.create_chat_completion.return_value = expected_response

    messages = [{"role": "user", "content": "Hello"}]
    response = backend.chat_completion(messages)

    assert response == expected_response
    mock_llm.create_chat_completion.assert_called_once_with(
        messages=messages,
        tools=None,
        tool_choice="auto",
        response_format=None,
        temperature=0.7,
        max_tokens=None,
        stop=None,
    )


def test_str_representation(sample_config, mock_model, mock_llama):
    """Test string representation."""
    backend = LlamaCppBackend(sample_config)

    assert "LlamaCppBackend" in str(backend)
    assert "test-model.gguf" in str(backend)
