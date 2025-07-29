import json

from ada.formatter import dump
from ada.logger import build_logger

logger = build_logger(__name__)

NULL_OUTPUT = "DERP"


class Response:
    """
    Parses the LLM source object and returns text
    """

    source: dict
    content: str  # keep in its original string format
    body: str
    role: str = "assistant"
    tokens: int = 0  # number of tokens used in the response

    def __init__(self, source: dict):
        logger.info("initialising response with \n" + dump(source))
        self.source = source
        self.tokens = source.get("usage", {}).get("total_tokens", 0)
        self.content, self.body = self.parse()

    def choose(self) -> dict:
        return self.source["choices"][0]

    def maybe_json(self, content) -> dict | str:
        try:
            parsed = json.loads(content)
            logger.info("content parsed as json")
            return parsed
        except json.JSONDecodeError:
            logger.info("content treated as a string")
            return content

    def parse(self) -> tuple[dict, str]:
        try:
            choice = self.choose()
            content = choice["message"]["content"]

            parsed = self.maybe_json(content)
            if isinstance(parsed, str):
                # cooerce string to dict just to simplify downstream processing
                return json.dumps({"text": parsed}), parsed
            elif isinstance(parsed, dict):
                return content, self.format(parsed)
            else:
                raise TypeError("unexpected type for content")
        except Exception as e:
            logger.error(e)
            logger.error("unable to parse llm source")
            logger.error("\n" + dump(self.source))
            return {}, NULL_OUTPUT

    def format(self, parsed) -> str:
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
            output.append(f"```\n{parsed['code']}\n```")

        if len(output) == 0:
            logger.error(f"unable to extract keys {output.keys()}")
            logger.info("\n" + dump(parsed))

            return NULL_OUTPUT

        return "\n\n".join(str(o) for o in output)
