class Persona:
    """
    A persona represents a specific AI assistant personality and behavior.
    Each persona has a name, a description, and a system prompt that defines its characteristics.
    """

    def __init__(self, name: str, description: str = "", prompt: str = ""):
        self.name = name
        self.description = description
        self.prompt = prompt

    def __str__(self) -> str:
        return f"{self.name}: {self.description}"

    def __repr__(self) -> str:
        return f"Persona(name='{self.name}', description='{self.description}', prompt='{self.prompt}')"
