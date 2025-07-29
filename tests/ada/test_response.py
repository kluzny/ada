import json

from tests.helpers.fixtures import parse

from ada.response import Response


def test_response():
    trivial = parse("llama/trivial.json")
    Response(trivial)


def test_response_usage():
    trivial = parse("llama/trivial.json")
    response = Response(trivial)

    assert response.tokens == 69


def test_response_can_parse_trivial_content():
    trivial = parse("llama/trivial.json")
    response = Response(trivial)

    assert response.body == "foo"
    assert response.role == "assistant"
    assert response.content == json.dumps({"text": "foo"})


def test_response_can_parse_simple_content():
    simple = parse("llama/simple.json")
    response = Response(simple)

    assert response.body == "bar"
    assert response.role == "assistant"
    assert response.content == json.dumps({"result": "bar"})


def test_response_can_parse_list_content():
    list = parse("llama/list.json")
    response = Response(list)

    assert response.body == "['Madrid', 'Barcelona', 'Valencia', 'Seville', 'Zaragoza']"
