from llama_cpp import Llama

from config import Config
from model import Model

class Agent:
  model: Model

  def __init__(self):
    config = Config()
    self.model = Model(config.model_url())
    self.llm = self.build_llm()

  def say(self, input: str) -> None:
    print(f"ADA: {input}")

  def build_llm(self):
    return Llama(
      model_path=self.model.path,
      n_ctx=2048,
      n_threads=4,
      verbose=False
    )

  def chat(self):
    print("🧠 LLM Chat (type 'exit' to quit)")
    while True:
        prompt = input("User: ")
        if prompt.lower() == "exit":
            break
        output = self.llm(prompt, max_tokens=256, stop=["\n", "User:"])
        print("ADA:", output["choices"][0]["text"].strip())

if __name__ == "__main__":
  agent = Agent()
  agent.say("Hello World!")
  agent.chat()