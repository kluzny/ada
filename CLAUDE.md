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

For GPU inference with CUDA:
```bash
CMAKE_ARGS="-DGGML_CUDA=on" uv pip install llama-cpp-python --verbose
```

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

### Key Patterns

1. **Asyncio Architecture**: Uses TaskGroup for concurrent file watching and chat interaction
2. **Pydantic Models**: Core data structures (Conversation, Entry, Response) use Pydantic for validation
3. **Hot Reloading**: Persona memories are watched via watchdog and trigger prompt rebuilds
4. **LLM Response Format**: Enforces JSON responses with optional keys ["text", "code"]
5. **Logging**: Uses custom logger (ada/logger.py) that logs to files in `logs/` directory by default

### Directory Structure

```
ada/
  agent.py           # Main agent orchestrator
  persona.py         # Persona class with memory loading
  personas.py        # Predefined persona definitions
  conversation.py    # Conversation history management
  model.py          # Model download and caching
  config.py         # Configuration loading
  tool_box.py       # Tool registry
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
models/            # Cached GGUF model files
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

## Configuration

Edit `config.json` to customize:
- `log_level`: DEBUG, INFO, WARNING, ERROR
- `record`: true/false to save conversations
- `history`: true/false for input history across sessions
- `model.url`: URL to GGUF model file
- `model.tokens`: Context window size (e.g., 2048)
