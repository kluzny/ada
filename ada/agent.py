from llama_cpp import Llama

from ada.config import Config
from ada.model import Model
from ada.conversation import Conversation
from ada.logger import build_logger
from ada.response import Response

WHOAMI = "ADA"
WHOAREYOU = "USER"

SYSTEM_PROMPT = """
You are an expert assistant named ADA.
Your primary task is answering USER queries.
Response concisely while returning critical information.
"""

logger = build_logger(__name__)

MAX_CONTEXT_LENGTH = 32768  # maximum context length for the LLM


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
            n_ctx=MAX_CONTEXT_LENGTH,
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
                response = Response(self.think(messages=self.conversation.messages()))

                if response.tokens >= 0.75 * MAX_CONTEXT_LENGTH:
                    logger.warning(
                        f"tokens {response.tokens} exceed 75% of max {MAX_CONTEXT_LENGTH}."
                    )

                self.conversation.append_response(WHOAMI, response)
                self.say(response.body)

    def muse(self, prompt: str):
        """
        think, but without the conversation context
        """
        output = self.llm(prompt, max_tokens=None, stop=[f"{WHOAREYOU}:"])
        return output["choices"][0]["text"].strip()

    def think(self, messages: list[dict] = []):
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

        return self.llm.create_chat_completion(
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.7,
            max_tokens=None,
            stop=[f"{WHOAREYOU}:"],
        )


if __name__ == "__main__":
    agent = Agent()
    agent.chat()
