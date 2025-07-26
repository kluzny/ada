class Entry:
    """
    A structured entry in a Conversation
    """

    author: str | None = None
    body: str | None = None

    def __init__(self, author: str, body: str):
        self.author = author
        self.body = body

    def __str__(self):
        return f"{self.author}: {self.body}"

    def __repr__(self):
        return f"Entry(author={self.author}, body={self.body})"
