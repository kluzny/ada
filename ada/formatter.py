"""
Output helpers
"""


def block(text: str, character: str = "*", length: int = 20):
    output = ""
    output += line()
    output += text.center(length, character) + "\n"
    output += line()
    return output


def line(character: str = "*", length: int = 20):
    return character * length + "\n"
