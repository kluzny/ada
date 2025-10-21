# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ADA (Agentic Digital Assistant) is a Python-based LLM agent framework that provides an interactive REPL for conversing with local language models via llama-cpp-python. The project emphasizes modularity through a persona system, extensibility through tools, and hot-reloadable context through memories.

## Development Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/):

```bash
# Initial setup
uv venv                    # create virtual environment
uv sync                    # install dependencies
cp -pv config.json.example config.json

# Development commands
make test                  # run all tests
make lint                  # format and fix issues (runs both format and fix)
make format                # format code with ruff
make fix                   # fix linting issues with ruff
make check                 # run lint + test
pytest                     # run all tests
pytest tests/ada/test_agent.py  # run specific test file
make clean                 # remove conversations and logs
make purge                 # clean + remove downloaded models

# Running
python main.py             # start the interactive REPL
```

The project uses an environment variable `APP_ENV=test` for testing (configured in pyproject.toml).

For GPU inference with CUDA (llama-cpp-python only):
```bash
CMAKE_ARGS="-DGGML_CUDA=on" uv pip install llama-cpp-python --verbose
```

## Backend Configuration

ADA supports two LLM backends: **llama-cpp-python** (local GGUF models) and **Ollama** (local model serving).

Configuration uses a `backend` key to select the active backend and a `backends` object containing backend-specific settings.

### Configuration Structure

Example `config.json` with both backends configured:
```json
{
  "log_level": "DEBUG",
  "record": true,
  "history": true,
  "backend": "llama-cpp",
  "backends": {
    "llama-cpp": {
      "model": "phi-2",
      "threads": 4,
      "verbose": false,
      "models": [
        {
          "name": "phi-2",
          "url": "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q8_0.gguf"
        }
      ]
    },
    "ollama": {
      "url": "http://localhost:11434",
      "model": "llama3.2"
    }
  }
}
```

### llama-cpp-python Backend

Runs GGUF models directly using llama.cpp. Models are automatically downloaded and cached in `models/` directory.
Context window size is automatically detected from the GGUF model metadata.

**Configuration keys:**
- `model`: Name of the model to use (references a model in the `models` array)
- `models`: Array of model definitions with:
  - `name`: Identifier for the model
  - `url`: Download URL for the GGUF file
- `threads`: Number of CPU threads (optional, default: 1)
- `verbose`: Enable verbose llama.cpp output (optional, default: false)

### Ollama Backend

Connects to a running Ollama instance. Requires Ollama to be installed and running separately (`ollama serve`).
Context window size is automatically detected from the Ollama model metadata.

**Configuration keys:**
- `url`: Ollama server URL (default: http://localhost:11434)
- `model`: Name of the Ollama model (e.g., "llama2", "llama3.2", "mistral")

**To switch backends:** Change the top-level `backend` key to either `"llama-cpp"` or `"ollama"`.

## Architecture

### Core Components

**Agent (ada/agent.py)**
- Main orchestrator that runs the interactive REPL
- Manages conversation loop, persona switching, and LLM interactions
- Uses asyncio TaskGroup for concurrent file watching and chat
- Special commands: `clear`, `history`, `tools`, `prompt`, `modes`/`mode`, `switch [name]`, `exit`
- System uses `WHOAMI = "ADA"` and `WHOAREYOU = "USER"` as role identifiers

**Persona System (ada/persona.py, ada/personas.py)**
- Personas define different AI assistant behaviors via system prompts
- Built-in personas: `default` (standard assistant), `jester` (responds in rhyme/jokes/pig-latin)
- Supports hot-reloadable "memories" - context files in `memories/[persona_name]/` loaded alphabetically
- Memories are wrapped in `<memory></memory>` tags and injected into system prompts
- File watcher automatically rebuilds persona when memory files change

**Conversation (ada/conversation.py)**
- Tracks conversation history as a list of Entry objects
- Optionally records to JSON in `conversations/` directory (controlled by config.record)
- Provides message formatting for LLM consumption

**Model (ada/model.py)**
- Handles downloading and caching GGUF models from URLs
- Downloads to `models/` directory with progress bar
- Model is configured via config.json (url and token context length)

**ToolBox (ada/tool_box.py, ada/tools/)**
- Extensible tool system for LLM function calling
- Tools inherit from `ada.tools.Base` abstract class
- Each tool defines name, description, parameters (JSON schema), and a `call()` method
- Tools are registered in `ToolBox.AVAILABLE_TOOLS` list
- See `ada/tools/example.py` for reference implementation

**Config (ada/config.py)**
- Loads from `config.json` (default) or custom path
- Settings: log_level, record (conversation saving), history (input history), model.url, model.tokens

**Backend System (ada/backends/)**
- **Base** - Abstract class defining the backend interface (chat_completion method)
- **LlamaCppBackend** - Runs local GGUF models via llama-cpp-python
- **OllamaBackend** - Connects to Ollama server for model inference
- Backends are selected in config.json via the `backend` field
- All backends return OpenAI-compatible response format for consistency

### Key Patterns

1. **Asyncio Architecture**: Uses TaskGroup for concurrent file watching and chat interaction
2. **Pydantic Models**: Core data structures (Conversation, Entry, Response) use Pydantic for validation
3. **Hot Reloading**: Persona memories are watched via watchdog and trigger prompt rebuilds
4. **LLM Response Format**: Enforces JSON responses with optional keys ["text", "code"]
5. **Backend Abstraction**: Pluggable backend system allows switching between llama-cpp and Ollama
6. **Logging**: Uses custom logger (ada/logger.py) that logs to files in `logs/` directory by default

### Directory Structure

```
ada/
  agent.py           # Main agent orchestrator
  persona.py         # Persona class with memory loading
  personas.py        # Predefined persona definitions
  conversation.py    # Conversation history management
  model.py          # Model download and caching (llama-cpp)
  config.py         # Configuration loading
  tool_box.py       # Tool registry
  backends/
    base.py         # Abstract base class for backends
    llama_cpp_backend.py  # llama-cpp-python implementation
    ollama_backend.py     # Ollama implementation
  tools/
    base.py         # Abstract base class for tools
    example.py      # Example tool implementation
  filesystem/
    async_file_watcher.py
    directory_watcher.py
memories/           # Persona-specific context files
  [persona_name]/
    001_*.txt       # Loaded alphabetically
conversations/      # Saved conversation JSON files
models/            # Cached GGUF model files (llama-cpp only)
logs/              # Application logs
tests/             # Test suite mirrors ada/ structure
```

## Testing

- Python 3.13+ required
- Tests use pytest with APP_ENV=test environment variable
- Test fixtures in `tests/helpers/fixtures.py`
- Tests mirror the source structure under `tests/ada/`

## Tool Development

To add a new tool:

1. Create a new class in `ada/tools/` inheriting from `Base`
2. Define `__init__` with name, description, and parameters (JSON schema)
3. Implement the `call()` method with tool logic
4. Import and add to `ToolBox.AVAILABLE_TOOLS` in `ada/tool_box.py`

Example:
```python
from ada.tools.base import Base

class MyTool(Base):
    def __init__(self):
        super().__init__(
            name="my_tool",
            description="Does something useful",
            parameters={
                "properties": {
                    "arg": {"type": "string", "description": "An argument"}
                }
            }
        )

    def call(self, arg: str, **kwargs) -> str:
        return f"Result: {arg}"
```

## Persona Development

1. Add persona definition to `ada/personas.py` as class attribute of `Personas`
2. Create directory in `memories/[persona_name]/`
3. Add numbered memory files (e.g., `001_context.txt`) - loaded alphabetically
4. Switch via REPL command: `switch [persona_name]`

## Backend Development

To add a new backend:

1. Create a new class in `ada/backends/` inheriting from `Base`
2. Implement the `chat_completion()` method following the OpenAI-compatible format
3. Add configuration handling in `Config.backend_config()`
4. Update `Agent.__build_backend()` to support the new backend type
5. Import and export in `ada/backends/__init__.py`

Example skeleton:
```python
from ada.backends.base import Base

class MyBackend(Base):
    def __init__(self, config: dict):
        super().__init__(config)
        # Initialize your backend

    def chat_completion(self, messages, tools=None, **kwargs) -> dict:
        # Call your backend API
        # Return OpenAI-compatible response format
        return {
            "choices": [{"message": {"role": "assistant", "content": "..."}}],
            "usage": {"total_tokens": 100}
        }
```

## Configuration

Edit `config.json` to customize:

**Top-level settings:**
- `log_level`: DEBUG, INFO, WARNING, ERROR
- `record`: true/false to save conversations to JSON files
- `history`: true/false for input history across sessions
- `backend`: "llama-cpp" or "ollama" - selects which backend to use

**Backend-specific settings (under `backends` object):**

For `llama-cpp`:
- `model`: Name of model to use from the `models` array
- `models`: Array of model definitions with `name` and `url`
- `threads`: Number of CPU threads (optional, default: 1)
- `verbose`: Enable verbose output (optional, default: false)

For `ollama`:
- `url`: Ollama server URL (optional, default: http://localhost:11434)
- `model`: Name of the Ollama model (e.g., "llama2", "llama3.2", "mistral")