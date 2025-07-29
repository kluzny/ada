from llama_cpp import Llama

from ada.config import Config
from ada.model import Model
from ada.conversation import Conversation
from ada.logger import build_logger
from ada.answer import Answer

WHOAMI = "ADA"
WHOAREYOU = "USER"

SYSTEM_PROMPT = """
You are an expert assistant named ADA.
Your primary task is answering USER queries.
Answer concisely while returning critical information.
"""

logger = build_logger(__name__)


class Agent:
    """
    An interactive llm agent
    """

    model: Model
    conversation: Conversation

    def __init__(self):
        logger.info("initializing agent")
        config = Config()
        self.model = Model(config.model_url())
        self.llm = self.build_llm()
        self.conversation = Conversation()

    def say(self, input: str) -> None:
        print(f"{WHOAMI}: {input}")

    def build_llm(self):
        return Llama(
            model_path=self.model.path,
            # n_ctx=2048,
            n_ctx=32768,
            n_threads=4,
            verbose=False,  # TODO: can we capture the verbose output with our logger?
        )

    def chat(self):
        print(f"{WHOAMI} Chat (type 'exit' to quit)")
        while True:
            prompt = input(f"{WHOAREYOU}: ")
            if prompt.strip() == "":
                continue
            elif prompt.lower() == "exit":
                self.say("Goodbye")
                break
            elif prompt.lower() == "history":
                print(self.conversation)
            elif prompt.lower() == "clear":
                self.conversation.clear()
            else:
                self.conversation.append(WHOAREYOU, prompt)
                thought = self.think(prompt)
                self.conversation.append(WHOAMI, thought)
                self.say(thought)

    def muse(self, prompt: str):
        """
        Think, but without the conversation context
        """
        output = self.llm(prompt, max_tokens=None, stop=[f"{WHOAREYOU}:"])
        return output["choices"][0]["text"].strip()

    def think(self, prompt: str):
        output = self.thought(prompt)
        answer = Answer(output)
        return answer.parse()

    def thought(self, prompt: str):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.append({"role": "user", "content": prompt})

        return self.llm.create_chat_completion(
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.7,
            max_tokens=256,  # TODO: None
            stop=[f"{WHOAREYOU}:"],
        )


if __name__ == "__main__":
    agent = Agent()
    agent.chat()
