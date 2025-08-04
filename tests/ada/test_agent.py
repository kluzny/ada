from ada import Agent

TEST_CONFIG_PATH = "tests/fixtures/config/example.json"


def test_agent():
    Agent(config_path=TEST_CONFIG_PATH)


def test_agent_say():
    agent = Agent(config_path=TEST_CONFIG_PATH)
    agent.say("Hello World!")
