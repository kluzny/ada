"""
Tools submodule for Ada.

This module provides base classes and utilities for creating tools
that can be used by the Ada agent system.
"""

from .base import Base as Base
from .example import ExampleTool

__all__ = ["ExampleTool"]
