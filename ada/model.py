import os
import urllib.request

class Model:
  CACHE_DIR = "models"

  url = None
  name = None
  path = None

  def __init__(self, url: str):
    self.url = url
    self.name = url.split("/")[-1]
    print(f"MODEL: Using {self.name}")
    self.path = os.path.join(self.CACHE_DIR, self.name)

    self.prepare()

  def prepare(self):
    os.makedirs(self.CACHE_DIR, exist_ok=True)

    if not os.path.exists(self.path):
        print(f"MODEL: Downloading from {self.url}...")
        urllib.request.urlretrieve(self.url, self.path)
        print(f"MODEL: Saved to {self.path}")
    else:
        print(f"MODEL: Exists at {self.path}")