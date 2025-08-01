from llama_cpp import Llama

from ada.config import Config
from ada.model import Model
from ada.conversation import Conversation
from ada.logger import build_logger
from ada.response import Response
from ada.personas import Personas

WHOAMI = "ADA"
WHOAREYOU = "USER"

logger = build_logger(__name__)


class Agent:
    """
    An interactive llm agent
    """

    model: Model
    max_content_length: int
    conversation: Conversation

    def __init__(self):
        logger.info("initializing agent")
        config = Config()
        self.model = Model(config.model_url())
        self.max_content_length = config.model_tokens()
        self.llm = self.build_llm()
        self.conversation = Conversation()

    def say(self, input: str) -> None:
        print(f"{WHOAMI}: {input}")

    def build_llm(self):
        return Llama(
            model_path=self.model.path,
            n_ctx=self.max_content_length,
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

                logger.info(f"using {response.tokens} tokens")
                if response.tokens >= 0.75 * self.max_content_length:
                    logger.warning(
                        f"usage exceed 75% of max {self.max_content_length}."
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
        messages.insert(0, {"role": "system", "content": Personas.DEFAULT.prompt})

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
