from peft import PeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch

def main(lang="julia", target_epoch=4):
    print(f"Loading {lang} model for epoch {target_epoch}")
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        dtype=torch.bfloat16, 
        device_map="auto"
    )
    steps_per_epoch_dict = {
        "julia": 180,
        "lua": 28,
        "ocaml": 43,
        "r": 99,
        "racket": 95,
    }
    if target_epoch > 0:
        steps_per_epoch = steps_per_epoch_dict[lang]
        steps = steps_per_epoch * target_epoch
        peft_model_name = f"ckpt/pt/{lang}-manuals/checkpoint-{steps}"
        model = PeftModelForCausalLM.from_pretrained(model, peft_model_name)
    model.eval()

    streamer = TextStreamer(tokenizer)

    prompts = [
'''""" Given a positive floating point number, it can be decomposed into
and integer part (largest integer smaller than given number) and decimals
(leftover part always smaller than 1).
Return the decimal part of the number.
>>> truncate_number(3.5)
0.5"""
function truncate_number(number::Float64)::Float64''',
    ]

    def format_prompt(prompt: str) -> str:
        return f"Using given examples and the signature, generate the missing implementation in {lang} by wrapping your code in ```{lang.lower()} markdown blocks:\n\n{prompt}\n\n"

    for prompt in prompts:
        content = format_prompt(prompt)
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer(text, return_tensors="pt", padding = True).to(model.device)
        generated_ids = model.generate(**model_inputs, max_new_tokens=16384, do_sample=False, streamer=streamer)
        print("=" * 40)

if __name__ == "__main__":
    lang = "julia"
    epochs = [0, 2, 4]
    for epoch in epochs:
        main(lang, epoch)
