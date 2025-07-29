from tests.helpers.fixtures import parse

from ada.answer import Answer


def test_answer():
    Answer({})


def test_answer_can_parse_trivial_content():
    response = parse("llama/trivial.json")
    answer = Answer(response)
    assert answer.parse() == "foo"


def test_answer_can_parse_simple_content():
    response = parse("llama/simple.json")
    answer = Answer(response)
    assert answer.parse() == "bar"
