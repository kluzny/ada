from prompt_toolkit import prompt, PromptSession
from prompt_toolkit.history import FileHistory

from textwrap import dedent

from asyncio import TaskGroup, Queue, AbstractEventLoop, to_thread

from ada.config import Config
from ada.conversation import Conversation
from ada.logger import build_logger
from ada.response import Response
from ada.persona import Persona
from ada.personas import Personas
from ada.tool_box import ToolBox
from ada.exceptions import TerminateTaskGroup
from ada.looper import Looper
from ada.formatter import block
from ada.backends import Base as Backend, LlamaCppBackend, OllamaBackend
from ada.voice import Voice


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
        self.config = config
        self.backend: Backend = self.__build_backend(config)
        self.max_content_length: int = self.backend.context_window()
        logger.info(f"max_content_length: {self.max_content_length}")
        self.conversation: Conversation = Conversation(record=config.record())
        self.persona = Personas.DEFAULT
        if config.voice():
            self.voice = Voice(config.voice())  # pyright: ignore[reportArgumentType] not bool under if
            self.voice.say("Hello World!")
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

        if self.config.voice():
            self.voice.say(input)

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

    def __build_backend(self, config: Config) -> Backend:
        """
        Build the appropriate LLM backend based on configuration.

        Args:
            config: Configuration object

        Returns:
            Initialized backend instance
        """
        backend = config.backend()
        backend_config = config.backend_config()

        logger.info(f"building backend: {backend}")

        if backend == "llama-cpp":
            return LlamaCppBackend(backend_config)
        elif backend == "ollama":
            return OllamaBackend(backend_config)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def __show_help(self) -> None:
        """Display available commands and their descriptions."""
        help_text = dedent("""\
        Available Commands:

        /help, /?           - Show this help message
        /clear              - Clear the conversation history
        /history            - Display the conversation history
        /tools              - List available tools
        /prompt             - Show the current system prompt
        /persona, /personas - Show current persona and list available personas
        /switch [name]      - Switch to a different persona
        /backend, /backends - Show current backend and list available backends
        /model, /models     - Show current model and list available models
        /exit, /quit, /bye  - Exit the chat
        """)

        self.say(help_text)

    def __show_backends(self) -> None:
        """Display current backend and list available backends."""
        current_backend = self.config.backend()
        available_backends = list(self.config.loaded.get("backends", {}).keys())

        output = f"Current backend: {current_backend}\n"
        output += f"Backend type: {self.backend}\n\n"
        output += "Available backends:\n\n"
        for backend in available_backends:
            marker = "*" if backend == current_backend else " "
            output += f"  {marker} {backend}\n"

        self.say(output)

    def __show_models(self) -> None:
        """Display current model and list available models."""
        current_model = self.backend.current_model()
        available_models = self.backend.available_models()

        output = f"Current model: {current_model}\n\n"
        output += "Available models:\n\n"
        for model in available_models:
            marker = "*" if model == current_model else " "
            output += f"  {marker} {model}\n"

        self.say(output)

    async def __scan_commands(self, query: str, looper: Looper) -> bool:
        """
        Scan for and handle special commands.

        Args:
            query: The user input to scan for commands

        Returns:
            bool: True if a command was handled, False if no command was found
        """
        neat = query.lower().strip()
        if neat == "/help" or neat == "/?":
            self.__show_help()
            return True
        elif neat == "/clear":
            self.conversation.clear()
            return True
        elif neat == "/history":
            print(self.conversation)
            return True
        elif neat == "/tools":
            self.__list_tools()
            return True
        elif neat == "/prompt":
            self.say(
                "\n"
                + block("SYSTEM PROMPT")
                + self.__system_prompt()["content"]
                + "\n"
                + block("END SYSTEM PROMPT").strip()
            )
            return True
        elif neat == "/personas" or neat == "/persona":
            current = f"Current persona is:\n\n{self.persona}\n"
            available_personas = ""
            for persona in Personas.all():
                available_personas += str(persona) + "\n"
            self.say(
                f"{current}\nAvailable personas:\n\n{available_personas}\nUse `/switch [name]` to change personas."
            )
            return True
        elif query.lower().startswith("/switch "):
            persona_name = query[8:].strip()  # Remove "/switch " prefix
            switched = await self.__switch_persona(persona_name, looper)
            if switched:
                self.say(f"Switched to persona {self.persona.name}")
            else:
                self.say(
                    f"Persona '{persona_name}' not found. Use '/personas' to see available personas."
                )
            return True
        elif neat == "/backends" or neat == "/backend":
            self.__show_backends()
            return True
        elif neat == "/models" or neat == "/model":
            self.__show_models()
            return True
        return False

    def __process_message(self, query: str):
        """
        Process a user message and generate a response.

        Args:
            query: The user's input message
        """
        self.conversation.append(WHOAREYOU, query)
        thought = self.__think(messages=self.conversation.messages())
        response = Response(thought)

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
        print(f"{WHOAMI} Chat (type '/exit' to quit, '/help' for commands)")

        while True:
            query = await to_thread(lambda: self.input(f"{WHOAREYOU}: "))
            if query.strip() == "":
                continue  # ignore empty user input
            elif await self.__scan_commands(query, looper):
                continue  # command was handled by __scan_commands
            elif query.lower().strip() in ("/exit", "/quit", "/bye"):
                self.say("Goodbye")
                break
            else:
                self.__process_message(query)

        raise TerminateTaskGroup

    def __think(self, messages: list[dict] | None = None) -> dict:
        if messages is None:
            messages = []
        messages.insert(0, self.__system_prompt())

        return self.backend.chat_completion(
            messages=messages,
            tools=ToolBox.definitions(),
            # tool_choice={
            #     "type": "function",
            #     "function": {"name": "example_tool"},
            # },
            tool_choice="auto",
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
