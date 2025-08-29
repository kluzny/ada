"""
Base class for Ada tools.

This module provides the Base class that all tools should inherit from.
"""

from abc import ABC, abstractmethod
from typing import Optional, Callable, Any


class Base(ABC):
    """
    Base class for all Ada tools.

    This class provides a common interface for tools that can be used
    by the Ada agent system. Subclasses should override the call method
    to implement their specific functionality.
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize the Base tool.

        Args:
            name: The name of the tool
            description: A description of what the tool does
            parameters: A dictionary defining the tool's parameters
        """
        self.name = name
        self.description = description
        self.parameters = parameters or {}

    @abstractmethod
    def call(self, *args, **kwargs) -> Any:
        """
        Execute the tool's main functionality.

        This method should be overridden by subclasses to implement
        the specific tool behavior.

        Args:
            *args: Positional arguments for the tool
            **kwargs: Keyword arguments for the tool

        Returns:
            The result of the tool execution
        """
        raise NotImplementedError("Subclasses must implement the call method")

    def definition(self) -> dict[str, Any]:
        """
        Get the tool's definition in the standard format.

        Returns:
            A dictionary containing the tool's definition in the standard format
        """
        properties = self.parameters.get("properties", {})
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": list(properties.keys()),
                },
            },
        }

    def create_global_function(self) -> Callable:
        """
        Create a global function for a tool instance.

        Returns:
            A callable function that can be used to call the tool
        """

        def tool_function(*args, **kwargs):
            """
            Global function to call a tool.

            This function is used by the LLM to call tools.
            All arguments and keyword arguments are passed directly to the tool's call method.

            Args:
                *args: Positional arguments for the tool
                **kwargs: Keyword arguments for the tool

            Returns:
                The result of the tool execution
            """
            return self.call(*args, **kwargs)

        # Set the function name and docstring
        tool_function.__name__ = self.name
        tool_function.__doc__ = f"Global function to call the {self.name} tool."

        return tool_function

    def __params(self) -> str:
        definition = self.definition()
        params = definition["function"]["parameters"]["required"]
        return ", ".join(params)

    def __str__(self) -> str:
        return f"{self.name}({self.__params()}): {self.description}"
