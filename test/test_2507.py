from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-4B-Thinking-2507"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# prepare the model input
prompts = ["Give me a short introduction to large language model.",
           "Do you know about Junsoo Byun who is from Korea?"]
messages_list = [[{"role": "user", "content": prompt}] for prompt in prompts]

texts = [
    tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    for messages in messages_list
]

model_inputs = tokenizer(texts, return_tensors="pt", padding = True).to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=16384
)

# 출력 결과 분리
for i, input_ids in enumerate(model_inputs['input_ids']):
    output = generated_ids[i][len(input_ids):]  # 새로 생성된 토큰만
    try:
        # </think> 토큰 ID 찾아서 분리
        index = len(output) - output[::-1].tolist().index(151668)
    except ValueError:
        index = 0

    thinking = tokenizer.decode(output[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output[index:], skip_special_tokens=True).strip("\n")
    print(f"[Prompt {i+1}]")
    print("thinking content:", thinking)
    print("content:", content)
    print("=" * 40)
