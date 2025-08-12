"""
Example tool implementation.

This module demonstrates how to create a tool by inheriting from the Base class.
"""

from .base import Base


class ExampleTool(Base):
    """
    An example tool that demonstrates the Base class usage.

    This tool simply returns a greeting message with the provided name.
    """

    def __init__(self):
        """Initialize the example tool with predefined name, description, and parameters."""
        name = "example_tool"
        description = "An example tool that returns a greeting"
        parameters = {
            "properties": {
                "name": {"type": "string", "description": "The name to greet"}
            }
        }
        super().__init__(name, description, parameters)

    def call(self, name: str, **kwargs) -> str:
        """
        Execute the example tool.

        Args:
            name: The name to greet
            **kwargs: Additional keyword arguments (ignored)

        Returns:
            A greeting message
        """
        return f"Hello, {name}! This is an example tool."
