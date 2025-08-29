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

Note: `llama-cpp-python` also supports GPU inference with additional compiler flags, for instance:

```bash
# building with CUDA support

$ CMAKE_ARGS="-DGGML_CUDA=on" uv pip install llama-cpp-python --verbose
```

Refer to the official documentation for your specific GPU [llama-cpp-python#installation](https://llama-cpp-python.readthedocs.io/en/latest/#installation)

## Running

```bash
$ python main.py # a minimal REPL for the agent
```

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
