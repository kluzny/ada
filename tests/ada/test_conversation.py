import os
import json
import tempfile

from pathlib import Path
from ada.conversation import Conversation
from ada.entry import Entry
from ada.response import Response


def test_conversation():
    Conversation()


def test_conversation_initialization():
    """Test that Conversation initializes correctly"""
    conversation = Conversation()
    assert conversation.history == []
    assert isinstance(conversation.history, list)
    assert not conversation.record
    assert conversation.storage_path is None
    assert conversation.record_path is None


def test_conversation_initialization_record_enabled():
    """Test that Conversation initializes file paths when recording"""
    with tempfile.TemporaryDirectory() as temp_dir:
        conversation = Conversation(record=True, storage_path=temp_dir)
        assert conversation.record
        assert conversation.storage_path is not None
        assert conversation.record_path is not None


def test_conversation_append():
    """Test appending a simple entry to conversation"""
    conversation = Conversation()
    conversation.append("USER", "Hello, how are you?")

    assert len(conversation.history) == 1
    entry = conversation.history[0]
    assert isinstance(entry, Entry)
    assert entry.author == "USER"
    assert entry.body == "Hello, how are you?"
    assert entry.role == "user"
    assert entry.content is None


def test_conversation_append_response():
    """Test appending a response to conversation"""
    # Create a mock response source
    response_source = {
        "choices": [{"message": {"content": "I'm doing well, thank you for asking!"}}],
        "usage": {"total_tokens": 10},
    }

    response = Response(response_source)
    conversation = Conversation()
    conversation.append_response("ASSISTANT", response)

    assert len(conversation.history) == 1
    entry = conversation.history[0]
    assert entry.author == "ASSISTANT"
    assert entry.body == "I'm doing well, thank you for asking!"
    assert entry.role == "assistant"
    assert entry.content == '{"text": "I\'m doing well, thank you for asking!"}'


def test_conversation_append_response_with_json():
    """Test appending a response with JSON content"""
    # Create a mock response source with JSON content
    json_content = json.dumps({"text": "Here's the answer", "code": "print('hello')"})
    response_source = {
        "choices": [{"message": {"content": json_content}}],
        "usage": {"total_tokens": 15},
    }

    response = Response(response_source)
    conversation = Conversation()
    conversation.append_response("ASSISTANT", response)

    assert len(conversation.history) == 1
    entry = conversation.history[0]
    assert entry.author == "ASSISTANT"
    assert entry.body == "Here's the answer\n\n```\nprint('hello')\n```"
    assert entry.role == "assistant"
    assert entry.content == json_content


def test_conversation_clear():
    """Test clearing conversation history"""
    conversation = Conversation()
    conversation.append("USER", "Hello")
    conversation.append("ASSISTANT", "Hi")

    assert len(conversation.history) == 2

    conversation.clear()
    assert len(conversation.history) == 0
    assert conversation.history == []


def test_conversation_messages():
    """Test getting messages from conversation"""
    conversation = Conversation()
    conversation.append("USER", "Hello")
    conversation.append("ASSISTANT", "Hi there!")

    messages = conversation.messages()

    assert len(messages) == 2
    assert messages[0] == {"role": "user", "content": "Hello"}
    assert messages[1] == {"role": "user", "content": "Hi there!"}


def test_conversation_messages_empty():
    """Test getting messages from empty conversation"""
    conversation = Conversation()

    messages = conversation.messages()

    assert messages == []
    assert len(messages) == 0


def test_conversation_str():
    """Test string representation of conversation"""
    conversation = Conversation()
    conversation.append("USER", "Hello")
    conversation.append("ASSISTANT", "Hi there!")

    conversation_str = str(conversation)

    assert "HISTORY START" in conversation_str
    assert "USER: Hello" in conversation_str
    assert "ASSISTANT: Hi there!" in conversation_str
    assert "HISTORY END" in conversation_str


def test_conversation_str_empty():
    """Test string representation of empty conversation"""
    conversation = Conversation()

    conversation_str = str(conversation)

    assert "HISTORY START" in conversation_str
    assert "HISTORY END" in conversation_str


def test_conversation_record_filename_format():
    """Test that record filename follows timestamp-uuid.json format"""
    with tempfile.TemporaryDirectory() as temp_dir:
        conversation = Conversation(record=True, storage_path=temp_dir)

        assert conversation.record_path is not None
        filename = os.path.basename(conversation.record_path)

        # Check format: timestamp-uuid.json
        parts = filename.replace(".json", "").split("-")
        assert len(parts) >= 2

        # First part should be a timestamp (numeric)
        assert parts[0].isdigit()

        # Last part should be a UUID (36 characters with hyphens)
        uuid_part = "-".join(parts[1:])
        assert len(uuid_part) == 36
        assert uuid_part.count("-") == 4


def test_conversation_record_json_creation():
    """Test that JSON file is created only when an entry is added"""
    with tempfile.TemporaryDirectory() as temp_dir:
        conversation = Conversation(record=True, storage_path=temp_dir)

        # File should not exist after initialization
        json_files = list(Path(temp_dir).glob("*.json"))
        assert len(json_files) == 0

        # Add an entry
        conversation.append("USER", "Hello")
        json_files = list(Path(temp_dir).glob("*.json"))
        assert len(json_files) == 1

        # Check that file contains the entry
        with open(json_files[0], "r") as f:
            data = json.load(f)
            assert len(data) == 1
            assert data[0]["author"] == "USER"
            assert data[0]["body"] == "Hello"


def test_conversation_record_json_updates():
    """Test that JSON file is updated when entries are added"""
    with tempfile.TemporaryDirectory() as temp_dir:
        conversation = Conversation(record=True, storage_path=temp_dir)
        # Add an entry
        conversation.append("USER", "Hello")

        json_file = list(Path(temp_dir).glob("*.json"))[0]
        # Check that JSON file was updated
        with open(json_file, "r") as f:
            data = json.load(f)
            assert len(data) == 1
            assert data[0]["author"] == "USER"
            assert data[0]["body"] == "Hello"

        # Add another entry
        conversation.append("ASSISTANT", "Hi there!")

        # Check that JSON file was updated again
        with open(json_file, "r") as f:
            data = json.load(f)
            assert len(data) == 2
            assert data[1]["author"] == "ASSISTANT"
            assert data[1]["body"] == "Hi there!"


def test_conversation_record_response_json_updates():
    """Test that JSON file is updated when responses are added"""
    with tempfile.TemporaryDirectory() as temp_dir:
        conversation = Conversation(record=True, storage_path=temp_dir)

        # Create a mock response
        response_source = {
            "choices": [{"message": {"content": "I'm doing well!"}}],
            "usage": {"total_tokens": 10},
        }
        response = Response(response_source)

        # Add a response
        conversation.append_response("ASSISTANT", response)

        json_file = list(Path(temp_dir).glob("*.json"))[0]

        # Check that JSON file was updated
        with open(json_file, "r") as f:
            data = json.load(f)
            assert len(data) == 1
            assert data[0]["author"] == "ASSISTANT"
            assert data[0]["body"] == "I'm doing well!"
            assert data[0]["role"] == "assistant"
            assert data[0]["content"] is not None


def test_conversation_record_clear_json_updates():
    """Test that JSON file is deleted when conversation is cleared"""
    with tempfile.TemporaryDirectory() as temp_dir:
        conversation = Conversation(record=True, storage_path=temp_dir)
        # Add some entries
        conversation.append("USER", "Hello")
        conversation.append("ASSISTANT", "Hi")
        json_file = list(Path(temp_dir).glob("*.json"))[0]

        # Clear conversation
        conversation.clear()

        # Check that JSON file was deleted
        assert not Path(json_file).exists()
