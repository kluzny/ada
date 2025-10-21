import pytest
from ada.backends.base import Base


@pytest.fixture
def sample_config():
    """Sample configuration for testing"""
    return {"test": "config"}


def test_base_is_abstract(sample_config):
    """Test that Base cannot be instantiated directly."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        Base(sample_config)  # pyright: ignore[reportAbstractUsage]


def test_base_requires_chat_completion(sample_config):
    """Test that subclasses must implement chat_completion."""

    class IncompleteBackend(Base):
        def current_model(self) -> str:
            return "test-model"

        def available_models(self) -> list[str]:
            return ["test-model"]

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        IncompleteBackend(sample_config)  # pyright: ignore[reportAbstractUsage]


def test_base_requires_current_model(sample_config):
    """Test that subclasses must implement current_model."""

    class IncompleteBackend(Base):
        def chat_completion(self, messages, **kwargs):
            return {}

        def available_models(self) -> list[str]:
            return ["test-model"]

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        IncompleteBackend(sample_config)  # pyright: ignore[reportAbstractUsage]


def test_base_requires_available_models(sample_config):
    """Test that subclasses must implement available_models."""

    class IncompleteBackend(Base):
        def chat_completion(self, messages, **kwargs):
            return {}

        def current_model(self) -> str:
            return "test-model"

        def context_window(self) -> int:
            return 2048

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        IncompleteBackend(sample_config)  # pyright: ignore[reportAbstractUsage]


def test_base_requires_context_window(sample_config):
    """Test that subclasses must implement context_window."""

    class IncompleteBackend(Base):
        def chat_completion(self, messages, **kwargs):
            return {}

        def current_model(self) -> str:
            return "test-model"

        def available_models(self) -> list[str]:
            return ["test-model"]

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        IncompleteBackend(sample_config)  # pyright: ignore[reportAbstractUsage]


def test_complete_backend_implementation(sample_config):
    """Test that a complete implementation can be instantiated."""

    class CompleteBackend(Base):
        def chat_completion(self, messages, **kwargs):
            return {
                "choices": [{"message": {"role": "assistant", "content": "test"}}],
                "usage": {"total_tokens": 10},
            }

        def current_model(self) -> str:
            return "test-model"

        def available_models(self) -> list[str]:
            return ["test-model", "another-model"]

        def context_window(self) -> int:
            return 4096

    backend = CompleteBackend(sample_config)

    # Test that config is stored
    assert backend.config == sample_config

    # Test that all methods work
    assert backend.current_model() == "test-model"
    assert backend.available_models() == ["test-model", "another-model"]
    assert backend.context_window() == 4096

    response = backend.chat_completion([{"role": "user", "content": "test"}])
    assert "choices" in response
    assert "usage" in response


def test_base_str_method():
    """Test the __str__ method."""

    class TestBackend(Base):
        def chat_completion(self, messages, **kwargs):
            return {}

        def current_model(self) -> str:
            return "test"

        def available_models(self) -> list[str]:
            return []

        def context_window(self) -> int:
            return 2048

    backend = TestBackend({})
    assert str(backend) == "TestBackend"


def test_chat_completion_signature():
    """Test that chat_completion has correct signature."""

    class TestBackend(Base):
        def chat_completion(
            self,
            messages,
            tools=None,
            tool_choice="auto",
            response_format=None,
            temperature=0.7,
            max_tokens=None,
            stop=None,
        ):
            # Verify all parameters are passed
            return {
                "messages": messages,
                "tools": tools,
                "tool_choice": tool_choice,
                "response_format": response_format,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stop": stop,
            }

        def current_model(self) -> str:
            return "test"

        def available_models(self) -> list[str]:
            return []

        def context_window(self) -> int:
            return 2048

    backend = TestBackend({})

    messages = [{"role": "user", "content": "Hello"}]
    tools = [{"type": "function", "function": {"name": "test"}}]
    response_format = {"type": "json_object"}

    result = backend.chat_completion(
        messages=messages,
        tools=tools,
        tool_choice="auto",
        response_format=response_format,
        temperature=0.8,
        max_tokens=100,
        stop=["STOP"],
    )

    assert result["messages"] == messages
    assert result["tools"] == tools
    assert result["tool_choice"] == "auto"
    assert result["response_format"] == response_format
    assert result["temperature"] == 0.8
    assert result["max_tokens"] == 100
    assert result["stop"] == ["STOP"]


def test_config_storage():
    """Test that config is properly stored in __init__."""

    class TestBackend(Base):
        def chat_completion(self, messages, **kwargs):
            return {}

        def current_model(self) -> str:
            return "test"

        def available_models(self) -> list[str]:
            return []

        def context_window(self) -> int:
            return 2048

    config = {
        "model": "test-model",
        "url": "http://localhost:8000",
        "custom_param": "value",
    }

    backend = TestBackend(config)

    assert backend.config == config
    assert backend.config["model"] == "test-model"
    assert backend.config["url"] == "http://localhost:8000"
    assert backend.config["custom_param"] == "value"


def test_methods_exist():
    """Test that all expected methods exist and are callable."""
    assert hasattr(Base, "__init__")
    assert hasattr(Base, "chat_completion")
    assert hasattr(Base, "current_model")
    assert hasattr(Base, "available_models")
    assert hasattr(Base, "context_window")
    assert hasattr(Base, "__str__")

    assert callable(getattr(Base, "__init__"))
    assert callable(getattr(Base, "chat_completion"))
    assert callable(getattr(Base, "current_model"))
    assert callable(getattr(Base, "available_models"))
    assert callable(getattr(Base, "context_window"))
    assert callable(getattr(Base, "__str__"))
