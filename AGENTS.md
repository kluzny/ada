# AGENTS.md

This file provides guidance to agents when working with code in this repository.

**IMPORTANT** Read [agents/rules.md](agents/rules.md) immediatley for additional rules that must always be followed when working with this codebase.
**IMPORTANT** Read [agents/style.md](agents/style.md) when producing source code for style, formatting and linting guidelines.

## Project Overview

ADA (Agentic Digital Assistant) is a Python-based LLM agent framework that provides an interactive REPL for conversing with local language models via llama-cpp-python. The project emphasizes modularity through a persona system, extensibility through tools, and hot-reloadable context through memories.

## Project Documentation

- **Implementation Plan**: See [docs/plan.md](docs/plan.md) for the detailed roadmap
- **Working List**: See [docs/todo.md](docs/todo.md) for current tasks and priorities
- **Known Issues**: See [docs/bugs.md](docs/bugs.md) for tracked bugs

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
- File watcher (watchdog) automatically detects changes and triggers persona rebuild
- Hot-reload applies immediately without restarting the REPL or losing conversation state
- Changes typically detected within 1-2 seconds of file modification

**Conversation (ada/conversation.py)**
- Tracks conversation history as a list of Entry objects
- Optionally records to JSON in `conversations/` directory (controlled by config.record)
- Provides message formatting for LLM consumption

**Model (ada/model.py)**
- Handles downloading and caching GGUF models from URLs
- Downloads to `.ada/llms/` directory with progress bar
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

**Core Data Models (Pydantic)**
- **Entry** (ada/entry.py) - Individual conversation entries with author and body fields
- **Response** (ada/response.py) - LLM response structure with optional text/code fields
- **Looper** (ada/looper.py) - Container for asyncio primitives (TaskGroup, EventLoop, Queue)
- **Conversation** (ada/conversation.py) - Manages list of Entry objects with message formatting

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
.ada/              # Artifact cache directory (auto-created)
  llms/            # Cached GGUF model files (llama-cpp only)
  voices/          # Cached TTS voice models (Piper)
logs/              # Application logs
tests/             # Test suite mirrors ada/ structure
```

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
pytest tests/ada/test_agent.py::test_agent_say  # run single test function
pytest -v                  # run tests with verbose output
pytest -k "test_name"      # run tests matching pattern
make clean                 # remove conversations and logs
make purge                 # clean + remove downloaded models

# Running
python main.py             # start the interactive REPL
```

### Make Commands Quick Reference

| Command | Description |
|---------|-------------|
| `make test` | Run all tests with pytest |
| `make format` | Format code with ruff |
| `make fix` | Auto-fix linting issues with ruff |
| `make lint` | Run both format and fix |
| `make check` | Run lint + test |
| `make clean` | Remove conversations and logs |
| `make purge` | Clean + remove downloaded models |

The project uses an environment variable `APP_ENV=test` for testing (configured in pyproject.toml).

For GPU inference with CUDA (llama-cpp-python only):
```bash
CMAKE_ARGS="-DGGML_CUDA=on" uv pip install llama-cpp-python --verbose
```

## Artifact Storage

ADA uses a `.ada/` directory (defined in `ada/constants.py` as `ARTIFACT_DIR`) for caching downloaded artifacts:

- `.ada/llms/` - Downloaded GGUF model files (llama-cpp backend)
- `.ada/voices/` - Downloaded TTS voice models (Piper TTS)

This directory is separate from:
- `conversations/` - Saved conversation JSON files
- `logs/` - Application log files
- `memories/` - Persona-specific context files

**Important**: The `.ada/` directory is created automatically and should be added to `.gitignore` to avoid committing large model files.

## Configuration

Edit `config.json` to customize:

**Top-level settings:**
- `log_level`: DEBUG, INFO, WARNING, ERROR
- `record`: true/false to save conversations to JSON files
- `history`: true/false for input history across sessions
- `backend`: "llama-cpp" or "ollama" - selects which backend to use

### Backend Configuration

ADA supports two LLM backends: **llama-cpp-python** (local GGUF models) and **Ollama** (local model serving).

Configuration uses a `backend` key to select the active backend and a `backends` object containing backend-specific settings.

#### Configuration Structure

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

#### llama-cpp-python Backend

Runs GGUF models directly using llama.cpp. Models are automatically downloaded and cached in `.ada/llms/` directory.
Context window size is automatically detected from the GGUF model metadata.

**Configuration keys:**
- `model`: Name of the model to use (references a model in the `models` array)
- `models`: Array of model definitions with:
  - `name`: Identifier for the model
  - `url`: Download URL for the GGUF file
- `threads`: Number of CPU threads (optional, default: 1)
- `verbose`: Enable verbose llama.cpp output (optional, default: false)

#### Ollama Backend

Connects to a running Ollama instance. Requires Ollama to be installed and running separately (`ollama serve`).
Context window size is automatically detected from the Ollama model metadata.

**Configuration keys:**
- `url`: Ollama server URL (default: http://localhost:11434)
- `model`: Name of the Ollama model (e.g., "llama2", "llama3.2", "mistral")

**To switch backends:** Change the top-level `backend` key to either `"llama-cpp"` or `"ollama"`.

## Testing

- Python 3.13+ required
- Tests use pytest with APP_ENV=test environment variable
- Test fixtures in `tests/helpers/fixtures.py`
- Tests mirror the source structure under `tests/ada/`
- See [agents/rules.md](agents/rules.md) for testing rules and best practices

## Tool System

### Tool Calling Flow

The tool system enables LLMs to call Python functions through a structured workflow:

1. **Tool Registration**: Tools are discovered and registered in `ToolBox.AVAILABLE_TOOLS` list
2. **LLM receives tools**: During chat completion, available tools are passed to the backend
3. **LLM requests tool call**: LLM includes tool call in response (OpenAI format)
4. **Agent executes tool**: Agent parses tool call and invokes the appropriate tool's `call()` method
5. **Result returned to LLM**: Tool output is formatted and added to conversation history

### Adding a New Tool

1. Create a new file in `ada/tools/` (e.g., `ada/tools/my_tool.py`)
2. Inherit from `ada.tools.Base` abstract class
3. Implement required methods:
   - `name` property: Unique tool identifier
   - `description` property: What the tool does (used by LLM)
   - `parameters` property: JSON schema defining expected arguments
   - `call()` method: Tool implementation that accepts `**kwargs`
4. Register in `ada/tool_box.py` by adding to `ToolBox.AVAILABLE_TOOLS` list

**Example tool structure:**
```python
from ada.tools.base import Base
from typing import Any

class MyTool(Base):
    @property
    def name(self) -> str:
        return "my_tool"
    
    @property
    def description(self) -> str:
        return "Does something useful"
    
    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "arg_name": {"type": "string", "description": "What this arg does"}
            },
            "required": ["arg_name"]
        }
    
    def call(self, **kwargs) -> Any:
        arg_value = kwargs.get("arg_name")
        # Tool implementation here
        return {"result": "success"}
```

See `ada/tools/example.py` for a complete reference implementation.
