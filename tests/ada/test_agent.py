from ada import Agent
from ada.config import Config

TEST_CONFIG_PATH = "tests/fixtures/config/example.json"


def test_agent():
    config = Config(path=TEST_CONFIG_PATH)
    Agent(config=config)


def test_agent_say():
    config = Config(path=TEST_CONFIG_PATH)
    agent = Agent(config=config)
    agent.say("Hello World!")
