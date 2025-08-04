import json
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
