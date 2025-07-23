import requests
from typing import Dict
from google.generativeai.types import Tool

class OllamaModel:
    def __init__(self, model="mistral:latest"):
        self.url = "http://localhost:11434/api/generate"
        self.model = model

    def generate(self, prompt: str) -> str:
        response = requests.post(self.url, json={
            "model": self.model,
            "prompt": prompt,
            "stream": False
        })
        if response.ok:
            return response.json()["response"]
        else:
            raise Exception(f"Ollama error: {response.text}")



class LLMTool(Tool):
    def __init__(self, ollama: OllamaModel):
        self.ollama = ollama

    def call(self, input: Dict) -> Dict:
        prompt = input.get("prompt", "")
        result = self.ollama.generate(prompt)
        return {"response": result}