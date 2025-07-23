from typing import List
from llama_cpp import Llama

from ada.config import Config
from ada.model import Model


class Agent:
    model: Model
    conversation: List[str] = []

    def __init__(self):
        config = Config()
        self.model = Model(config.model_url())
        self.llm = self.build_llm()

    def say(self, input: str) -> None:
        print(f"ADA: {input}")

    def build_llm(self):
        return Llama(model_path=self.model.path, n_ctx=2048, n_threads=4, verbose=False)

    def chat(self):
        print("🧠 LLM Chat (type 'exit' to quit)")
        while True:
            prompt = input("User: ")
            if prompt.lower() == "exit":
                self.say("Goodbye")
                break
            elif prompt.lower() == "history":
                self.print_history()
            elif prompt.lower() == "clear":
                self.clear_history()
            else:
                self.conversation.append(f"User: {prompt}")
                thought = self.think(prompt)

                self.conversation.append(f"ADA: {thought}")
                self.say(thought)

    def think(self, prompt: str):
        output = self.llm(prompt, max_tokens=256, stop=["\n", "User:"])
        return output["choices"][0]["text"].strip()

    def print_block(self, text: str, length: int = 20):
        print("*" * length)
        print(text.center(length, "*"))
        print("*" * length)

    def print_history(self):
        self.print_block("HISTORY START")
        print(*self.conversation, sep="\n")
        self.print_block("HISTORY END")

    def clear_history(self):
        self.conversation = []
        print("HISTORY: Cleared")


if __name__ == "__main__":
    agent = Agent()
    agent.say("Hello World!")
    agent.chat()
