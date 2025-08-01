class Persona:
    """
    A persona represents a specific AI assistant personality and behavior.
    Each persona has a name and a system prompt that defines its characteristics.
    """

    def __init__(self, name: str, prompt: str):
        self.name = name
        self.prompt = prompt

    def __str__(self) -> str:
        return f"Persona({self.name})"

    def __repr__(self) -> str:
        return f"Persona(name='{self.name}', prompt='{self.prompt}')"
