import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
import gc
gc.collect()
from transformers import pipeline
import torch
torch.cuda.empty_cache()

model_id = "openai/gpt-oss-20b"

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype="auto",
    device_map="auto",
)

messages = [
    {"role": "user", "content": "Explain quantum mechanics clearly and concisely."},
]

outputs = pipe(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])