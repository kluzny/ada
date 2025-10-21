import json


from ada.formatter import dump
from ada.logger import build_logger
from ada.tool_box import ToolBox

logger = build_logger(__name__)

NULL_OUTPUT = "DERP"

# create all the tool methods for the Agent to call
for tool in ToolBox.tools:
    globals()[tool.name] = tool.create_global_function()


# TODO: switch to pydantic model
class Response:
    """
    Parses the LLM source object and returns text
    """

    source: dict
    content: str | None
    body: str
    role: str = "assistant"
    tokens: int = 0  # number of tokens used in the response

    def __init__(self, source: dict) -> None:
        logger.info("initialising response with \n" + dump(source))
        self.source = source
        self.tokens = source.get("usage", {}).get("total_tokens", 0)
        self.__parse()

    def __choose(self) -> dict:
        return self.source["choices"][0]

    def __maybe_json(self, content: str | None) -> dict | str:
        try:
            parsed = json.loads(content)
            logger.info("content parsed as json")
            return parsed
        except json.JSONDecodeError:
            logger.info("content treated as a string")
            return content

    def __parse(self) -> None:
        try:
            choice = self.__choose()
            message = choice["message"]
            raw_content = message["content"]

            if raw_content is not None:
                parsed_content = self.__maybe_json(raw_content)

                if isinstance(parsed_content, str):
                    # cooerce string to dict just to simplify downstream processing
                    content = json.dumps({"text": parsed_content})
                    body = parsed_content
                elif isinstance(parsed_content, dict):
                    content = raw_content
                    body = self.__format(parsed_content)
                else:
                    raise TypeError("unexpected type for content")
            else:
                content = None
                body = ""

            if "tool_calls" in message:
                body += self.__handle_tool_calls(message["tool_calls"])

            self.content = content
            self.body = body
        except Exception as e:
            logger.error(e)
            logger.error("unable to parse llm source")
            logger.error("\n" + dump(self.source))
            self.content = raw_content
            self.body = NULL_OUTPUT

    def __handle_tool_calls(self, tool_calls: list[dict]) -> str:
        returns: list[str] = []

        tool_functions = [
            tool_call for tool_call in tool_calls if tool_call.get("type") == "function"
        ]

        for tool_function in tool_functions:
            function_signature = tool_function["function"]
            function_name = function_signature["name"]
            keyword_args = json.loads(function_signature["arguments"])

            logger.info(f"invoking {function_name} with {keyword_args}")
            function = globals()[function_name]
            returns.append(function(**keyword_args))

        return "\n".join(returns)

    def __format(self, parsed: dict) -> str:
        output = []

        if "text" in parsed:
            output.append(parsed["text"])

        if "answer" in parsed:
            output.append(parsed["answer"])

        if "result" in parsed:
            output.append(parsed["result"])

        if "message" in parsed:
            output.append(parsed["message"])

        if "output" in parsed:
            output.append(parsed["output"])

        if "code" in parsed:
            if parsed["code"] is not None and parsed["code"].strip() != "":
                output.append(f"```\n{parsed['code']}\n```")

        if len(output) == 0:
            logger.error(f"unable to extract keys {parsed.keys()}")
            logger.info("\n" + dump(parsed))

            return NULL_OUTPUT

        return "\n\n".join(str(o) for o in output)
