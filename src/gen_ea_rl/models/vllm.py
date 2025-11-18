from instructor import patch, Mode, from_provider
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI
from typing import Optional
import os
import logging

# logging.basicConfig(level=logging.DEBUG)
os.environ['OPENAI_API_KEY'] = 'token-abc123'
os.environ['MISTRAL_API_KEY'] = 'token-abc123'

class vllm:
    model_name: str
    reasoning_effort: Optional[str] = "high"
    temperature: float = 1.0
    top_p: Optional[float] = None
    top_logprobs: Optional[int] = None

    def __init__(self, model_name: str = "openai/gpt-oss-20b"):
        if "gpt-oss-20b" in model_name:
            mode = Mode.JSON_SCHEMA
        else:
            mode = Mode.TOOLS
        self.client = patch(OpenAI(base_url="http://localhost:8000/v1"), mode=mode, max_retries=5)
        self.model_name = model_name

    def __call__(self, **kwargs):
        mode = kwargs.get('mode', None)
        prompt = kwargs.get('prompt', None)

        messages = []
        if hasattr(mode, "developer"):
            messages.append({"role": "developer", "content": mode.developer()})
        messages.append({"role": "user", "content": prompt})

        output = self.client.chat.completions.create(
            model=self.model_name,
            response_model=mode,
            reasoning_effort=self.reasoning_effort,
            temperature=self.temperature,
            top_p=self.top_p,
            logprobs= True if self.top_logprobs is not None else False,
            top_logprobs=self.top_logprobs,
            messages=messages,
        )

        return output
