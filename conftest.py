import pytest
from ada.backends.ollama_backend import OllamaBackend


def pytest_collection_modifyitems(config, items):
    ollama_items = [item for item in items if item.get_closest_marker("ollama")]
    if not ollama_items:
        return

    ollama_available = OllamaBackend.is_available()

    for item in ollama_items:
        if not ollama_available:
            item.add_marker(
                pytest.mark.skip(reason="Ollama is not available on this system")
            )
            continue

        marker = item.get_closest_marker("ollama")
        model = marker.kwargs.get("model")
        if model and not OllamaBackend.is_model_available(model):
            item.add_marker(
                pytest.mark.skip(
                    reason=f"Ollama model '{model}' is not available on this system"
                )
            )
