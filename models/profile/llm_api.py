import os

from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

load_dotenv()

WAIT_TIME_MIN = 1
WAIT_TIME_MAX = 3
MAX_RETRIES = 3


def get_api_key() -> str:
    key = os.environ.get("OpenRouter_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Missing OpenRouter_API_KEY in environment.")
    return key


class LLMCaller:
    def __init__(self, platform: str, model_name: str):
        platform_norm = str(platform).strip()
        if platform_norm != "OpenRouter":
            raise ValueError("Only OpenRouter platform is supported.")

        self.model_dict = {
            "qwen-2.5-7b": "qwen2.5-7b-instruct",
            "deepseek-v3": "deepseek-ai/DeepSeek-V3",
            "gpt-3.5-turbo": "gpt-3.5-turbo",
            "gpt-4o-mini": "gpt-4o-mini",
            "emb-3s": "text-embedding-3-small",
            "emb-ada2": "text-embedding-ada-002",
        }

        if model_name not in self.model_dict:
            raise ValueError(f"Unsupported model alias: {model_name}")

        self.model_name = self.model_dict[model_name]
        self.client = OpenAI(api_key=get_api_key(), base_url="https://openrouter.ai/api/v1")

    @retry(wait=wait_random_exponential(min=WAIT_TIME_MIN, max=WAIT_TIME_MAX), stop=stop_after_attempt(MAX_RETRIES))
    def get_response(self, messages, max_tokens: int = 1024, temperature: float = 0.0, return_json=None):
        params = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if return_json:
            params["response_format"] = {"type": "json_object"}
        completion = self.client.chat.completions.create(**params)
        return completion.choices[0].message.content.strip()

    @retry(wait=wait_random_exponential(min=WAIT_TIME_MIN, max=WAIT_TIME_MAX), stop=stop_after_attempt(MAX_RETRIES))
    def get_embedding(self, text: str):
        text = text.replace("\n", " ")
        response = self.client.embeddings.create(
            model=self.model_name,
            dimensions=128,
            input=text,
        )
        return response.data[0].embedding
