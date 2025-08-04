from llama_cpp import Llama

from ada.config import Config
from ada.model import Model
from ada.conversation import Conversation
from ada.logger import build_logger
from ada.response import Response
from ada.persona import Persona
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
    persona: Persona

    # TODO: inject config, rather than pass in the config path
    def __init__(self, config_path: str = None):
        logger.info("initializing agent")
        config = Config(path=config_path)
        self.model = Model(config.model_url())
        self.max_content_length = config.model_tokens()
        self.llm = self.build_llm()
        self.conversation = Conversation(record=config.record())
        self.persona = Personas.DEFAULT
        logger.info(f"using {self.persona.name} persona")

    def say(self, input: str) -> None:
        print(f"{WHOAMI}: {input}")

    def switch_persona(self, name: str) -> bool:
        """
        Switch to a different persona by name.

        Args:
            name: The name of the persona to switch to

        Returns:
            bool: True if switch was successful, False otherwise
        """
        persona = Personas.get(name)
        if persona is None:
            logger.warning(
                f"Persona '{name}' not found. Available personas: {[p.name for p in Personas.all()]}"
            )
            return False

        self.persona = persona
        logger.info(f"Switched to persona: {persona}")
        return True

    def build_llm(self):
        return Llama(
            model_path=self.model.path,
            n_ctx=self.max_content_length,
            n_threads=4,
            verbose=False,  # TODO: can we capture the verbose output with our logger?
        )

    def scan_commands(self, prompt: str) -> bool:
        """
        Scan for and handle special commands.

        Args:
            prompt: The user input to scan for commands

        Returns:
            bool: True if a command was handled, False if no command was found
        """
        neat = prompt.lower().strip()
        if neat == "clear":
            self.conversation.clear()
            return True
        elif neat == "history":
            print(self.conversation)
            return True
        elif neat == "modes" or neat == "mode":
            current = f"Current mode is:\n\n{self.persona}\n"
            available_personas = ""
            for persona in Personas.all():
                available_personas += str(persona) + "\n"
            self.say(
                f"{current}\nAvailable personas:\n\n{available_personas}\nUse `switch [name]` to change personas."
            )
            return True
        elif prompt.lower().startswith("switch "):
            persona_name = prompt[7:].strip()  # Remove "switch " prefix
            if self.switch_persona(persona_name):
                self.say(f"Switched to persona: {self.persona.name}")
            else:
                self.say(
                    f"Persona '{persona_name}' not found. Use 'modes' to see available personas."
                )
            return True
        return False

    def process_message(self, prompt: str):
        """
        Process a user message and generate a response.

        Args:
            prompt: The user's input message
        """
        self.conversation.append(WHOAREYOU, prompt)
        response = Response(self.think(messages=self.conversation.messages()))

        logger.info(f"using {response.tokens} tokens")
        if response.tokens >= 0.75 * self.max_content_length:
            logger.warning(f"usage exceed 75% of max {self.max_content_length}.")

        self.conversation.append_response(WHOAMI, response)
        self.say(response.body)

    def chat(self):
        print(f"{WHOAMI} Chat (type 'exit' to quit)")
        while True:
            prompt = input(f"{WHOAREYOU}: ")
            if prompt.strip() == "":
                continue  # ignore empty user input
            elif self.scan_commands(prompt):
                continue  # command was handled by scan_commands
            elif prompt.lower() == "exit":
                self.say("Goodbye")
                break
            else:
                self.process_message(prompt)

    def muse(self, prompt: str):
        """
        think, but without the conversation context
        """
        output = self.llm(prompt, max_tokens=None, stop=[f"{WHOAREYOU}:"])
        return output["choices"][0]["text"].strip()

    def system_prompt(self) -> dict:
        return {
            "role": "system",
            "content": self.persona.prompt,
        }

    def think(self, messages: list[dict] = []):
        messages.insert(0, self.system_prompt())

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
