from typing import List

from ada.tools import Base, ExampleTool


class ToolBox:
    AVAILABLE_TOOLS = [ExampleTool]

    tools: List[Base] = [tool() for tool in AVAILABLE_TOOLS]
