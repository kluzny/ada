# ADA

Agentic Digital Assistant

## Installation

Project dependencies are managed via [uv](https://docs.astral.sh/uv/)

```bash
$ git clone git@github.com:kluzny/ada.git # clone the repository
$ cd ada # change directory
$ uv venv # initialize a virtual environment
$ uv sync # install python packages
$ cp -pv config.json.example config.json # default configuration file
```

See the section on backends to determine what additional dependencies you may need to install.

## Running

```bash
$ python main.py # a minimal REPL for the agent
```

## Backends

ADA supports multiple LLM backends for flexible model deployment. You can switch between backends by changing the `backend` key in your `config.json`.

### llama-cpp (Local GGUF Models)

The **llama-cpp** backend runs quantized GGUF models locally using [llama-cpp-python](https://github.com/abetlen/llama-cpp-python). This backend is ideal for running models entirely offline with full privacy.

**Features:**

- Runs models locally without internet connection (after initial download)
- Automatic model download and caching to `models/` directory
- Support for CPU and GPU inference (with CUDA/Metal/etc.)
- Dynamic context window detection from GGUF metadata
- No external service dependencies

**Prerequisites:**

Note: `llama-cpp-python` also supports GPU inference with additional compiler flags, for instance:

```bash
# building with CUDA support

$ CMAKE_ARGS="-DGGML_CUDA=on" uv pip install llama-cpp-python --verbose
```

Refer to the official documentation for your specific GPU [llama-cpp-python#installation](https://llama-cpp-python.readthedocs.io/en/latest/#installation)

**Configuration:**

```json
{
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
    }
  }
}
```

**Configuration Options:**

- `model`: Name of the model to use (must match a name in the `models` array)
- `models`: Array of model definitions with `name` and `url` fields
- `threads`: Number of CPU threads to use (default: 4)
- `verbose`: Enable verbose llama.cpp logging (default: false)

**Finding Models:**

Browse quantized GGUF models on [Hugging Face](https://huggingface.co/), search for models with the "GGUF" tag.

### ollama (Local Model Server)

The **ollama** backend connects to a running [Ollama](https://ollama.ai/) server for model inference. Ollama provides an easy-to-use model management system with a simple CLI.

**Features:**

- Easy model management via `ollama pull <model-name>`
- Supports the full Ollama model library, although you will want to select models with tool support
- Automatic context window detection from model metadata
- Can connect to local or remote Ollama servers
- Built-in model quantization and optimization

**Prerequisites:**

Install and start Ollama:

```bash
# Install Ollama (see https://ollama.com)
$ curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
$ ollama pull gpt-oss:latest

# Start the Ollama server (if not managed via systemd)
$ ollama serve
```

**Configuration:**

```json
{
  "backend": "ollama",
  "backends": {
    "ollama": {
      "url": "http://localhost:11434",
      "model": "gpt-oss:latest"
    }
  }
}
```

**Configuration Options:**

- `url`: Ollama server URL (default: http://localhost:11434)
- `model`: Name of the Ollama model to use (e.g., "llama3.2:latest", "mistral:latest")

**Available Models:**
Browse the [Ollama model library](https://ollama.ai/library) for available models. Popular options include:

- `gemma3` - Google's single gpu Gemini model
- `gpt-oss` - OpenAI's open source model
- `llama3.2` - Meta's Llama 3.2 model
- `mistral` - Mistral AI's base model
- `phi` - Microsoft's efficient small model

### Switching Backends

To switch between backends, simply update the `backend` field in your `config.json`:

```json
{
  "backend": "ollama", // Change this to "llama-cpp" or "ollama"
  "backends": {
    // Both backend configurations can coexist
  }
}
```

Both backend configurations can remain in your config file - only the active backend specified by the `backend` field will be used.

## Personas

Ada ships with some default personas that define system prompts for various modes of operation.

- **default** - A standard chat assistant with no specific customizations
- **jester** - The **default**, but trained poorly, as a joke.

You can use the `mode [name]` command to change between personas or simply `modes` to see all of the available personas.

## Memories

You can add custom context to the various personas using the `memories` directory. Files are organized by persona name and loaded alphabetically after the core system prompt. Memories use the `system` role. The exact `system` behaviour is determined by the specific model you are using. Memories are read as plain text, but your LLM may support text, markdown, json or other file formats. Memories are hot reloaded based on which persona is active.

```bash
# ./memories/default/001_important_stuff_to_always_place_in_context.txt
This project uses a python virtual environment that needs to be activated.
Preface all python commands with `source ./venv/bin/activate`.
```

```bash
# ./memories/default/002_something_that_loads_after_001.md
Always use python type hints.
Prefer using the native type hints instead of the `typing` module.
```

```bash
# ./memories/jester/001_use_an_uncomfortable_amount_of_puns.lol
Puns are the highest form of humor.
Use puns as much as possible.
Surround the puns with markdown for italics like *pun* .
It's ok to hallucinate in service of providing an additional *pun*portunity.
```

## Roadmap

- provide a checksum when the model is downloaded
- command autocomplete
- resumable/forkable conversations
- multi-step tasks
- better tool use using explicit `tool_choice` calls
- tools for file system access e.g. find, tree, cat, diff
- support advanced templating for memories, possibly jinja
- logging happens in a separate frame
