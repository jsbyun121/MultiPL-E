# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
model = AutoModelForCausalLM.from_pretrained(
    "openai/gpt-oss-20b", 
    torch_dtype="auto", 
    device_map="auto",
)
streamer = TextStreamer(tokenizer)
messages = [
    {"role": "user", "content": "How many r's are in the word strawberry?"},
]
inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
	reasoning_effort="medium",
).to(model.device)

print("\n\n\n=======================\n\n\n")
outputs = model.generate(**inputs, max_new_tokens=4096, streamer=streamer, eos_token_id=[200002, 199999])
print("\n\n\n=======================\n\n\n")
