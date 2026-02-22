import os
import pytest
from ada.backends.ollama_backend import OllamaBackend


def _mark_skip(message):
    return pytest.mark.skip(message)


def pytest_collection_modifyitems(config, items):
    ollama_items = [item for item in items if item.get_closest_marker("ollama")]
    if not ollama_items:
        return

    ollama_url = os.environ.get("OLLAMA_URL")

    for item in ollama_items:
        if not ollama_url:
            item.add_marker(
                _mark_skip("set OLLAMA_URL to run Ollama integration tests")
            )
            continue

        if not OllamaBackend.is_available(ollama_url):
            item.add_marker(_mark_skip(f"Ollama is not available at {ollama_url}"))
            continue

        marker = item.get_closest_marker("ollama")
        model = marker.kwargs.get("model")
        if model and not OllamaBackend.is_model_available(model, ollama_url):
            item.add_marker(
                _mark_skip(f"Ollama model '{model}' is not available at {ollama_url}")
            )
