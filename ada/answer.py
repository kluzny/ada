import json
from ada.formatter import dump
from ada.logger import build_logger

logger = build_logger(__name__)

NULL_OUTPUT = "DERP"


class Answer:
    """
    Parses the LLM response object and returns text
    """

    response: dict

    def __init__(self, response: dict):
        logger.info("initialising answer with \n" + dump(response))
        self.response = response

    def parse(self) -> str:
        """
        returns the text of the first choice from a chat json object
        """
        try:
            first_choice = self.response["choices"][0]
            content = first_choice["message"]["content"]
            return self.extract(content)
        except:  # noqa: E722
            logger.error("unable to parse llm response")
            logger.error("\n" + dump(self.response))

            return NULL_OUTPUT

    def maybe_json(self, content) -> dict | str:
        try:
            parsed = json.loads(content)
            logger.info("content parsed as json")
            return parsed
        except json.JSONDecodeError:
            logger.info("content treated as a string")
            return content

    def extract(self, content: str) -> str:
        logger.info("extracting content")

        parsed = self.maybe_json(content)

        if isinstance(parsed, str):
            return parsed
        elif isinstance(parsed, dict):
            return self.structure_output(parsed)
        else:
            logger.error(f"unhandled type {type(parsed)}")
            return NULL_OUTPUT

    def structure_output(self, parsed) -> str:
        output = []

        if "message" in parsed:
            output.append(parsed["message"])

        if "answer" in parsed:
            output.append(parsed["answer"])

        if "result" in parsed:
            output.append(parsed["result"])

        if "text" in parsed:
            output.append(parsed["text"])

        if "code" in parsed:
            output.append(f"```\n{parsed['code']}\n```")

        if len(output) == 0:
            logger.error(f"unable to extract keys {output.keys()}")
            logger.info("\n" + dump(parsed))

            return NULL_OUTPUT

        return "\n\n".join(output)
