from instructor import patch, Mode, from_provider
from openai import OpenAI
import os
os.environ['OPENAI_API_KEY'] = 'token-abc123'

class GptOss20BVLLM:
    model_name: str = "openai/gpt-oss-20b"
    reasoning_effort: str = "high"

    def __init__(self, model_name: str = "openai/gpt-oss-20b"):
        self.client = patch(OpenAI(base_url="http://localhost:8000/v1"), mode=Mode.JSON)
        self.model_name = model_name

    def __call__(self, **kwargs):
        mode = kwargs.get('mode', None)
        prompt = kwargs.get('prompt', None)

        output = self.client.chat.completions.create(
            model=self.model_name,
            response_model=mode,
            reasoning_effort=self.reasoning_effort,
            messages=[
                {"role": "developer", "content": mode.developer()},
                {"role": "user", "content": prompt},
            ],
        )

        return output
