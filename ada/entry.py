from pydantic import BaseModel


class Entry(BaseModel):
    """
    A structured entry in a Conversation
    """

    author: str
    body: str
    role: str = "user"
    content: str | None = None

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
