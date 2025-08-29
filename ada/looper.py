from asyncio import TaskGroup, AbstractEventLoop, Queue
from pydantic import BaseModel, ConfigDict


class Looper(BaseModel):
    """Holds references to a task group, event loop, and a queue"""

    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)

    tg: TaskGroup
    loop: AbstractEventLoop
    queue: Queue
