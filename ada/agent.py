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
from ada.looper import Looper
from ada.formatter import block


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
        self.llm: Llama = self.__build_llm()
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

    async def run(self, loop: AbstractEventLoop) -> None:
        logger.info("running")
        try:
            async with TaskGroup() as tg:
                looper = Looper(tg=tg, loop=loop, queue=Queue())
                self.__swap_persona(looper, Personas.DEFAULT)
                tg.create_task(self.__event_consumer(looper.queue))
                tg.create_task(self.__chat(looper))

            self.persona.unwatch()  # TODO: can we auto clean this up
        except ExceptionGroup as eg:
            if any(isinstance(e, TerminateTaskGroup) for e in eg.exceptions):
                logger.info("normal exit")
            else:
                logger.error(f"unhandled exception group: {eg}")
                raise
        finally:
            logger.info("stopping")

    def say(self, input: str) -> None:
        print(f"{WHOAMI}: {input}")

    async def __switch_persona(self, name: str, looper: Looper) -> bool:
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

        self.__swap_persona(looper, persona)
        return True

    def __build_llm(self):
        return Llama(
            model_path=self.model.path,
            n_ctx=self.max_content_length,
            n_threads=4,
            verbose=False,  # TODO: can we capture the verbose output with our logger?
        )

    async def __scan_commands(self, query: str, looper: Looper) -> bool:
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
            self.say(
                "\n"
                + block("SYSTEM PROMPT")
                + self.__system_prompt()["content"]
                + "\n"
                + block("END SYSTEM PROMPT").strip()
            )
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
            switched = await self.__switch_persona(persona_name, looper)
            if switched:
                self.say(f"Switched to persona {self.persona.name}")
            else:
                self.say(
                    f"Persona '{persona_name}' not found. Use 'modes' to see available personas."
                )
            return True
        return False

    def __process_message(self, query: str):
        """
        Process a user message and generate a response.

        Args:
            query: The user's input message
        """
        self.conversation.append(WHOAREYOU, query)
        response = Response(self.__think(messages=self.conversation.messages()))

        logger.info(f"using {response.tokens} tokens")
        if response.tokens >= 0.75 * self.max_content_length:
            logger.warning(f"usage exceed 75% of max {self.max_content_length}.")

        self.conversation.append_response(WHOAMI, response)
        self.say(response.body)

    async def __event_consumer(self, queue: Queue):
        """Consumes file system events."""
        while True:
            event_type, file_path = await queue.get()
            logger.info(f"Event: {event_type} - {file_path}")
            self.__rebuild_persona()
            queue.task_done()

    def __rebuild_persona(self) -> None:
        logger.info(f"rebuilding persona {self.persona.name}")
        self.persona.clear_cached_memories()

    async def __chat(self, looper: Looper):
        print(f"{WHOAMI} Chat (type 'exit' to quit)")

        while True:
            query = await to_thread(self.input, f"{WHOAREYOU}: ")
            if query.strip() == "":
                continue  # ignore empty user input
            elif await self.__scan_commands(query, looper):
                continue  # command was handled by __scan_commands
            elif query.lower() == "exit":
                self.say("Goodbye")
                break
            else:
                self.__process_message(query)

        raise TerminateTaskGroup

    def __think(self, messages: list[dict] = []):
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

    def __swap_persona(self, looper: Looper, persona: Persona) -> None:
        if self.persona is not None:
            self.persona.unwatch()

        logger.info(f"swapping to persona [{persona}]")
        self.persona = persona
        looper.tg.create_task(self.persona.watch(looper.loop, looper.queue))
