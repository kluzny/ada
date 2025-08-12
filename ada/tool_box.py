from typing import List

from ada.tools import Base, ExampleTool


class ToolBox:
    AVAILABLE_TOOLS = [ExampleTool]

    tools: List[Base] = [tool() for tool in AVAILABLE_TOOLS]

    @classmethod
    def definitions(cls) -> list[dict]:
        return [tool.definition() for tool in cls.tools]
