import json
from ada.entry import Entry


def test_entry():
    Entry("USER", "foobar")


def test_entry_initialization():
    entry = Entry("USER", "foobar")
    assert entry.author == "USER"
    assert entry.body == "foobar"
    assert entry.role == "user"
    assert entry.content is None


def test_entry_str():
    entry = Entry("USER", "foobar")
    assert str(entry) == "USER: foobar"


def test_entry_message():
    entry = Entry("USER", "foobar")
    assert entry.message() == {"role": "user", "content": "foobar"}


def test_entry_message_with_content():
    content = json.dumps({"text": "foobar"})

    entry = Entry(
        "USER",
        "foobar",
        role="assistant",
        content=content,
    )
    assert entry.message() == {
        "role": "assistant",
        "content": content,
    }
