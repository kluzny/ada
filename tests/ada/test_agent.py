from ada import Agent


def test_agent():
    Agent()


def test_agent_say():
    agent = Agent()
    agent.say("Hello World!")
