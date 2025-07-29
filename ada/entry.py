class Entry:
    """
    A structured entry in a Conversation
    """

    author: str
    body: str
    role: str
    content: dict | None

    def __init__(
        self,
        author: str,
        body: str,
        role: str = "user",
        content: dict | None = None,
    ):
        self.author = author
        self.body = body
        self.role = role
        self.content = content

    def message(self) -> dict:
        if self.content is None:
            content = self.body
        else:
            content = self.content

        return {
            "role": self.role,
            "content": content,
        }

    def __str__(self):
        return f"{self.author}: {self.body}"

    def __repr__(self):
        return f"Entry(author={self.author}, body={self.body}), role={self.role}, content={self.content})"
