import json

FIXTURE_PATH = "tests/fixtures/"


def parse(path: str) -> dict:
    with open(FIXTURE_PATH + path, "r") as f:
        return json.load(f)
