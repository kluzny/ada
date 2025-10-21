"""
Output helpers
"""

import json

DEFAULT_WIDTH = 80


def block(text: str, character: str = "*", length: int = DEFAULT_WIDTH) -> str:
    output = ""
    output += line()
    output += text.center(length, character) + "\n"
    output += line()
    return output


def line(character: str = "*", length: int = DEFAULT_WIDTH) -> str:
    return character * length + "\n"


def dump(data: dict) -> str:
    return json.dumps(data, sort_keys=True, indent=4)
