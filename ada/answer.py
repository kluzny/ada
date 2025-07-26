import json
from ada.formatter import dump
from ada.logger import build_logger

logger = build_logger(__name__)

NULL_OUTPUT = "DERP"


class Answer:
    """
    Parses the LLM response object and returns text
    """

    raw: dict

    def __init__(self, raw: dict):
        logger.info("initialising answer with \n" + dump(raw))
        self.raw = raw

    def parse(self) -> str:
        """
        returns the text of the first choice from a chat json object
        """
        try:
            first_choice = self.raw["choices"][0]
            content = first_choice["message"]["content"]
            return self.extract(content)
        except:  # noqa: E722
            logger.error("unable to parse answer")
            logger.error(dump(self.raw))

            return NULL_OUTPUT

    def extract(self, content: str) -> str:
        logger.info("extracting content")
        parsed = json.loads(content)
        logger.info("\n" + dump(parsed))

        output = []

        if "message" in parsed:
            output.append(parsed["message"])

        if "answer" in parsed:
            output.append(parsed["answer"])

        if "text" in parsed:
            output.append(parsed["text"])

        if "code" in parsed:
            output.append(f"```\n{parsed['code']}\n```")

        if len(output) == 0:
            logger.error(f"unable to extract keys {output.keys()}")
            return NULL_OUTPUT

        return "\n\n".join(output)
