from llama_cpp import Llama
from prompt_toolkit import prompt, PromptSession
from prompt_toolkit.history import FileHistory

from asyncio import TaskGroup, Queue, AbstractEventLoop, to_thread

from ada.config import Config
from ada.model import Model
from ada.conversation import Conversation
from ada.logger import build_logger
from ada.response import Response
from ada.persona import Persona
from ada.personas import Personas
from ada.tool_box import ToolBox
from ada.exceptions import TerminateTaskGroup

WHOAMI = "ADA"
WHOAREYOU = "USER"

logger = build_logger(__name__)


class Agent:
    """
    An interactive llm agent
    """

    HISTORY_FILE = ".ada_history"

    def __init__(self, config: Config):
        logger.info("initializing agent")
        self.model: Model = Model(config.model_url())
        self.max_content_length: int = config.model_tokens()
        self.llm: Llama = self.build_llm()
        self.conversation: Conversation = Conversation(record=config.record())
        self.persona = Personas.DEFAULT
        self.__init_prompt(config)

    def __init_prompt(self, config: Config) -> None:
        if config.history():
            logger.info(f"using history file: {self.HISTORY_FILE}")
            session = PromptSession(history=FileHistory(self.HISTORY_FILE))
            self.input = session.prompt
        else:
            self.input = prompt

    def say(self, input: str) -> None:
        print(f"{WHOAMI}: {input}")

    async def switch_persona(self, name: str) -> bool:
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

        logger.info(f"Switched to persona: {persona}")
        # TODO: has to switch to a queue message to the chat thread
        # await self.__activate_persona(persona)
        return True

    def build_llm(self):
        return Llama(
            model_path=self.model.path,
            n_ctx=self.max_content_length,
            n_threads=4,
            verbose=False,  # TODO: can we capture the verbose output with our logger?
        )

    def scan_commands(self, query: str) -> bool:
        """
        Scan for and handle special commands.

        Args:
            query: The user input to scan for commands

        Returns:
            bool: True if a command was handled, False if no command was found
        """
        neat = query.lower().strip()
        if neat == "clear":
            self.conversation.clear()
            return True
        elif neat == "history":
            print(self.conversation)
            return True
        elif neat == "tools":
            self.__list_tools()
            return True
        elif neat == "prompt":
            self.say("SYSTEM_PROMPT: " + self.__system_prompt()["content"])
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
        elif query.lower().startswith("switch "):
            persona_name = query[7:].strip()  # Remove "switch " prefix
            if self.switch_persona(persona_name):
                self.say(f"Switched to persona: {self.persona.name}")
            else:
                self.say(
                    f"Persona '{persona_name}' not found. Use 'modes' to see available personas."
                )
            return True
        return False

    def process_message(self, query: str):
        """
        Process a user message and generate a response.

        Args:
            query: The user's input message
        """
        self.conversation.append(WHOAREYOU, query)
        response = Response(self.think(messages=self.conversation.messages()))

        logger.info(f"using {response.tokens} tokens")
        if response.tokens >= 0.75 * self.max_content_length:
            logger.warning(f"usage exceed 75% of max {self.max_content_length}.")

        self.conversation.append_response(WHOAMI, response)
        self.say(response.body)

    async def event_consumer(self, queue: Queue):
        """Consumes file system events."""
        while True:
            event_type, file_path = await queue.get()
            logger.info(f"Event: {event_type} - {file_path}")
            queue.task_done()

    async def run(self, loop: AbstractEventLoop) -> None:
        logger.info("running")
        queue = Queue()

        async with TaskGroup() as tg:
            tg.create_task(self.event_consumer(queue))
            tg.create_task(self.activate_persona(Personas.DEFAULT, loop, queue))
            tg.create_task(self.chat())

        await self.persona.unwatch()  # TODO: can we auto clean this up

        logger.info("stopping")

    async def chat(self):
        print(f"{WHOAMI} Chat (type 'exit' to quit)")

        while True:
            query = await to_thread(self.input, f"{WHOAREYOU}: ")
            if query.strip() == "":
                continue  # ignore empty user input
            elif self.scan_commands(query):
                continue  # command was handled by scan_commands
            elif query.lower() == "exit":
                self.say("Goodbye")
                break
            else:
                self.process_message(query)

        raise TerminateTaskGroup

    def muse(self, query: str):
        """
        think, but without the conversation context
        """
        output = self.llm(query, max_tokens=None, stop=[f"{WHOAREYOU}:"])
        return output["choices"][0]["text"].strip()

    def think(self, messages: list[dict] = []):
        messages.insert(0, self.__system_prompt())

        return self.llm.create_chat_completion(
            messages=messages,
            tools=ToolBox.definitions(),
            # tool_choice={
            #     "type": "function",
            #     "function": {"name": "example_tool"},
            # },
            tool_choice="auto",  # not yet implemented https://github.com/abetlen/llama-cpp-python/issues/1338
            response_format={"type": "json_object"},
            temperature=0.7,
            max_tokens=None,
            stop=[f"{WHOAREYOU}:"],
        )

    def __system_prompt(self) -> dict:
        system_prompt = self.persona.get_prompt() + "\n"
        system_prompt += "Use any of the following tools:\n"
        system_prompt += "\n".join([str(tool) for tool in ToolBox.tools])

        return {
            "role": "system",
            "content": system_prompt.strip(),
        }

    def __list_tools(self) -> None:
        output = "Available tools:\n\n"
        output += "\n".join([str(tool) for tool in ToolBox.tools])
        self.say(output)

    async def activate_persona(
        self,
        persona: Persona,
        loop: AbstractEventLoop,
        queue: Queue,
    ) -> None:
        if self.persona is not None:
            await persona.unwatch()

        self.persona = persona
        logger.info(f"using {self.persona.name} persona")
        await self.persona.watch(loop, queue)
