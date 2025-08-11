from transformers import pipeline
import torch
from pprint import pprint

model_id = "openai/gpt-oss-20b"

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype="auto",
    device_map="auto",
)

messages = [
    {"role": "user", "content": "Explain about Seoul National University."},
]

outputs = pipe(
    messages,
    max_new_tokens=256,
)
print("="*20)
pprint(outputs)
print("="*20)
pprint(outputs[0])
print("="*20)
print(outputs[0]["generated_text"][-1])