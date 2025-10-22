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
- Automatic model download and caching to `.ada/llms/` directory
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

## Text-to-Speech (TTS)

ADA includes integrated text-to-speech capabilities powered by [Piper TTS](https://github.com/OHF-Voice/piper1-gpl), a high-quality open-source neural text-to-speech system.

**Features:**

- High-quality neural speech synthesis
- Streaming audio directly to speakers (no disk files)
- Real-time playback as audio is generated
- Extensive voice model library with multiple languages and genders
- Automatic model download and caching to `.ada/voices/` directory
- Fully offline operation (after initial voice model download)

**Configuration:**

Enable TTS by adding the `tts` option to your `config.json`:

```json
{
  "tts": "en_US-amy-medium"
}
```

**Available Voices:**

Piper provides voices in the format `<language>-<voice>-<quality>`. For the complete list of available voices, see the [Piper Voice Models](https://huggingface.co/rhasspy/piper-voices) collection.

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

- tools for file system access e.g. find, tree, cat, diff
- audio input STT, with wake word 'ada'
- multi-step tasks
- command autocomplete
- resumable/forkable conversations
- better tool use using explicit `tool_choice` calls
- support advanced templating for memories, possibly jinja
- provide a checksum when the model is downloaded
- logging happens in a separate frame
- gif/vid of Ada, doing the thing

## TODO

- Move the audio to a seperate thread, so that input can continue or queue.
- Consider how to interrupt ada to stop speaking. Perhaps a queue or bus we send a stop command to?
- Does voice synthesis take so much gpu resource, that we can't do inference. Does the next prompt in queue need to wait for speach to complete?
- Command to repeat the last thing Ada says, /repeat
- Audit all the agent say calls, to make sure Ada isn't 'talking' when she doesn't need to e.g. help output, menus, code blocks
- Response objects should have a .body method for all text, then a speakable method for text that should be spoken, again to not read code blocks
- Think about how to get better phonetic answers, for instance can we return phonemes alongside text in the llm output, could that simply be infered by an llm pass after?

## Bugs

### llama-cpp

- tool calling is still busted, and very model finicky
- auto-detecting the max content only works for a handful of models, perhaps it should have a configurable override
