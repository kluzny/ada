import json
from ada.entry import Entry


def test_entry():
    Entry(author="USER", body="foobar")


def test_entry_initialization():
    entry = Entry(author="USER", body="foobar")
    assert entry.author == "USER"
    assert entry.body == "foobar"
    assert entry.role == "user"
    assert entry.content is None


def test_entry_str():
    entry = Entry(author="USER", body="foobar")
    assert str(entry) == "USER: foobar"


def test_entry_message():
    entry = Entry(author="USER", body="foobar")
    assert entry.message() == {"role": "user", "content": "foobar"}


def test_entry_message_with_content():
    content = json.dumps({"text": "foobar"})

    entry = Entry(
        author="USER",
        body="foobar",
        role="assistant",
        content=content,
    )
    assert entry.message() == {
        "role": "assistant",
        "content": content,
    }


def tests_entry_dumps_to_json():
    content = json.dumps({"text": "foobar"})

    entry = Entry(
        author="USER",
        body="foobar",
        role="assistant",
        content=content,
    )

    assert (
        entry.model_dump_json()
        == '{"author":"USER","body":"foobar","role":"assistant","content":"{\\"text\\": \\"foobar\\"}"}'
    )
