from ada.tools import Base


class BaseImplementation(Base):
    """Concrete implementation of Base for testing purposes."""

    def call(self, *args, **kwargs):
        """Test implementation of the call method."""
        return "test_result"


def test_base_initialization_with_parameters():
    """Test Base initialization with name, description, and parameters."""
    name = "test_tool"
    description = "A test tool"
    parameters = {
        "properties": {"arg1": {"type": "string"}, "arg2": {"type": "integer"}}
    }
    base = BaseImplementation(name, description, parameters)

    assert base.name == name
    assert base.description == description
    assert base.parameters == parameters


def test_base_str():
    """Test the string represenation of a tool"""
    name = "test_tool"
    description = "A test tool"
    base = BaseImplementation(name, description)

    assert (str(base)) == "test_tool(): A test tool"


def test_base_str_with_parameters():
    """Test the string represenation of a tool"""
    name = "test_tool"
    description = "A test tool"
    parameters = {
        "properties": {"arg1": {"type": "string"}, "arg2": {"type": "integer"}}
    }
    base = BaseImplementation(name, description, parameters)

    assert (str(base)) == "test_tool(arg1, arg2): A test tool"


def test_base_initialization_without_parameters():
    """Test Base initialization without parameters."""
    name = "test_tool"
    description = "A test tool"
    base = BaseImplementation(name, description)

    assert base.name == name
    assert base.description == description
    assert base.parameters == {}


def test_definition_method():
    """Test the definition method returns the correct format."""
    name = "test_tool"
    description = "A test tool"
    parameters = {
        "properties": {"arg1": {"type": "string"}, "arg2": {"type": "integer"}}
    }
    base = BaseImplementation(name, description, parameters)
    definition = base.definition()

    expected = {
        "type": "function",
        "function": {
            "name": "test_tool",
            "description": description,
            "parameters": {
                "type": "object",
                "properties": parameters["properties"],
                "required": ["arg1", "arg2"],
            },
        },
    }
    assert definition == expected


def test_definition_method_without_parameters():
    """Test the definition method with empty parameters."""
    name = "test_tool"
    description = "A test tool"
    base = BaseImplementation(name, description)
    definition = base.definition()

    expected = {
        "type": "function",
        "function": {
            "name": "test_tool",
            "description": description,
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }
    assert definition == expected


def test_call_implementation():
    """Test that call method works in concrete implementation."""
    base = BaseImplementation("test_tool", "A test tool")
    result = base.call("test_arg")

    assert result == "test_result"


def test_create_global_function():
    """Test the create_global_function class method."""
    tool_instance = BaseImplementation("test_tool", "A test tool")
    global_function = tool_instance.create_global_function()

    # Test function name and docstring
    assert global_function.__name__ == "test_tool"
    assert "Global function to call the test_tool tool" in global_function.__doc__

    # Test function call
    result = global_function("test_arg")
    assert result == "test_result"
