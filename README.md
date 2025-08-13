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

CMAKE_ARGS="-DGGML_CUDA=on" uv pip install llama-cpp-python --verbose
```

Refer to the official documentation for your specific GPU [llama-cpp-python#installation](https://llama-cpp-python.readthedocs.io/en/latest/#installation)

## Running

```bash
$ python main.py # a minimal REPL for the Agent
```

## Roadmap

- command autocomplete
- resumable/forkable conversations
- multi-step tasks
- configurable system prompts
- better tool use using explicit `tool_choice` calls
- tools for file system access e.g. find, tree, cat, diff
