from llama_cpp import Llama

from ada.config import Config
from ada.model import Model
from ada.conversation import Conversation


class Agent:
    model: Model
    conversation: Conversation

    def __init__(self):
        config = Config()
        self.model = Model(config.model_url())
        self.llm = self.build_llm()
        self.conversation = Conversation()

    def say(self, input: str) -> None:
        print(f"ADA: {input}")

    def build_llm(self):
        return Llama(
            model_path=self.model.path,
            n_ctx=2048,
            n_threads=4,
            verbose=True,
        )

    def chat(self):
        print("🧠 ADA Chat (type 'exit' to quit)")
        while True:
            prompt = input("User: ")
            if prompt.lower() == "exit":
                self.say("Goodbye")
                break
            elif prompt.lower() == "history":
                print(self.conversation)
            elif prompt.lower() == "clear":
                self.conversation.clear()
            else:
                self.conversation.append("User", prompt)
                thought = self.think(prompt)
                self.conversation.append("ADA", thought)
                self.say(thought)

    def think(self, prompt: str):
        output = self.llm(prompt, max_tokens=None, stop=["\n", "User:"])
        return output["choices"][0]["text"].strip()


if __name__ == "__main__":
    agent = Agent()
    agent.chat()
