# Code Style Guidelines

This document defines the coding style, formatting, and linting standards for the ADA project.

## Python Version & Tools

### Python Version

- **Required**: Python 3.13+
- Use modern Python syntax (PEP 604 union types with `|`, not `Union`)

### Formatting & Linting

- **Tool**: Ruff (v0.12+) for formatting and linting
- **Commands**: `make format` (auto-format), `make fix` (auto-fix linting), `make lint` (both)

## Code Organization

### Import Style

```python
# Standard library imports first
import json
import logging
from pathlib import Path
from typing import Any

# Third-party imports second
from pydantic import BaseModel
from llama_cpp import Llama

# Local imports last (absolute imports from ada.*)
from ada.config import Config
from ada.logger import build_logger
from ada.backends.base import Base
```

### Naming Conventions

- **Classes**: PascalCase (e.g., `Agent`, `LlamaCppBackend`, `ToolBox`)
- **Functions/Methods**: snake_case (e.g., `build_logger`, `chat_completion`, `context_window`)
- **Constants**: SCREAMING_SNAKE_CASE (e.g., `WHOAMI`, `STORAGE_PATH`, `DEFAULT_PATH`)
- **Private methods**: Prefix with double underscore (e.g., `__init_storage_path`, `__save_record`)
- **Module-level variables**: snake_case (e.g., `logger = build_logger(__name__)`)

## Type System

### Type Hints

- **Required** for all function signatures (parameters and return types)
- Use modern union syntax: `str | None` not `Optional[str]`
- Use `dict[str, Any]` not `Dict[str, Any]`
- Use `list[Entry]` not `List[Entry]`
- Example:
```python
def append(self, author: str, body: str) -> None:
    ...

def messages(self) -> list[dict]:
    ...

def backend_config(self, backend: str | None = None) -> dict[str, Any]:
    ...
```

## Documentation

### Docstrings

- **Required** for classes, public methods, and modules
- Use triple-quoted strings with Google-style format:
```python
def chat_completion(
    self,
    messages: list[dict],
    tools: list[dict] | None = None,
) -> dict:
    """
    Generate a chat completion response.

    Args:
        messages: List of message dictionaries with 'role' and 'content'
        tools: Optional list of tool definitions for function calling

    Returns:
        A dictionary containing the response in OpenAI-compatible format

    Raises:
        ValueError: If configuration is invalid
    """
```

## Python Patterns

### Error Handling

- Use descriptive exception messages
- Raise `ValueError` for configuration/validation errors
- Custom exceptions inherit from `Exception` (see `ada/exceptions.py`)
- Example:
```python
if not model_name:
    raise ValueError("'model' is required in llama-cpp backend configuration")
```

### Pydantic Models

- Use Pydantic for data validation (see `Conversation`, `Entry`, `Response`)
- Use `Field()` with descriptions for model fields
- Override `__init__` when custom initialization logic is needed
```python
class Conversation(BaseModel):
    history: list[Entry] = Field(
        default_factory=list,
        description="List of conversation entries",
    )
```

### Async Patterns

- Use `asyncio.TaskGroup` for managing concurrent tasks
- Prefer `async with` for context management
- Use `to_thread()` for running blocking I/O in async context

### Logging

- Use the custom logger: `logger = build_logger(__name__)`
- Log levels: DEBUG for detailed flow, INFO for key events, WARNING for issues, ERROR for failures
- Example: `logger.info(f"initializing agent with model: {model_name}")`

### Abstract Base Classes

- Use `ABC` and `@abstractmethod` for base classes (see `ada/backends/base.py`, `ada/tools/base.py`)
- All abstract methods must be implemented by subclasses
- Example:
```python
from abc import ABC, abstractmethod

class Base(ABC):
    @abstractmethod
    def call(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Subclasses must implement the call method")
```
