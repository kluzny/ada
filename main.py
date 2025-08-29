import asyncio

from ada import Agent
from ada.config import Config


async def main():
    config = Config()
    agent = Agent(config=config)
    await agent.run(asyncio.get_running_loop())


if __name__ == "__main__":
    asyncio.run(main())
