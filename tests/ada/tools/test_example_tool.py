from ada.tools import ExampleTool


def test_example_tool():
    ExampleTool()


def test_example_tool_call():
    """Test ExampleTool call method."""
    tool = ExampleTool()
    result = tool.call("Alice")

    assert result == "Hello, Alice! This is an example tool."
