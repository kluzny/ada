"""
Output helpers
"""

import json


def block(text: str, character: str = "*", length: int = 20):
    output = ""
    output += line()
    output += text.center(length, character) + "\n"
    output += line()
    return output


def line(character: str = "*", length: int = 20):
    return character * length + "\n"


def dump(data: dict) -> str:
    return json.dumps(data, sort_keys=True, indent=4)
