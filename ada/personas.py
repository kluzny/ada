from ada.persona import Persona


class Personas:
    """
    Static collection of predefined personas.
    """

    DEFAULT = Persona(
        name="default",
        description="The standard expert assistant.",
        prompt="""
You are an expert assistant named ADA.
Your primary task is answering USER queries.
Respond concisely while returning critical information.
Respond only in json using the optional keys: ["text", "code"] or with available tool calls.
Only respond with code if prompted for source code.
""".strip(),
    )

    JESTER = Persona(
        name="jester",
        description="The standard expert assistant, but as a joke.",
        prompt=DEFAULT.prompt
        + """
Respond only in rhyme.
Occasionally, add a joke.
Occasionally, speak in pig-latin.
""".strip(),
    )

    @classmethod
    def all(cls) -> list[Persona]:
        """
        Dynamically list all persona constants defined in this class.

        Returns:
            list[Persona]: A list of all Persona instances defined as class attributes
        """
        personas = []
        for attr_name in dir(cls):
            # Skip private attributes and methods
            if attr_name.startswith("_"):
                continue

            attr_value = getattr(cls, attr_name)
            # Check if the attribute is a Persona instance
            if isinstance(attr_value, Persona):
                personas.append(attr_value)

        return personas

    @classmethod
    def get(cls, name: str) -> Persona | None:
        """
        Get a persona by its name.

        Args:
            name: The name of the persona to find

        Returns:
            Persona | None: The persona with the matching name, or None if not found
        """
        for persona in cls.all():
            if persona.name == name:
                return persona
        return None
