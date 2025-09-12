from unsloth import FastLanguageModel
from unsloth import UnslothTrainer, UnslothTrainingArguments
from datetime import datetime
import torch
import os
import argparse
max_seq_length = 4096

parser = argparse.ArgumentParser()
parser.add_argument("lang", choices = ["r", "julia", "lua", "racket", "ocaml"])
args = parser.parse_args()

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen3-4B-Instruct-2507", # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
    max_seq_length = max_seq_length,
    dtype = torch.bfloat16,
    load_in_4bit = False
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 128, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",
                      "embed_tokens", "lm_head",], # Add for continual pretraining
    lora_alpha = 32,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = True, # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = True,   # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    texts  = examples["text"]
    outputs = []
    for text in texts:
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        outputs.append(text + EOS_TOKEN)
    return {"text" : outputs}
pass

from datasets import load_dataset

dataset = load_dataset(f"jusjinuk/{args.lang}-manuals", split = "train")

# We select 1% of the data to make training faster!
dataset = dataset.train_test_split(train_size = 0.9)
dataset = dataset.map(formatting_prompts_func, batched = True)

# Prepare training arguments with optional wandb configuration
training_args = {
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "warmup_steps": 10,
    # warmup_ratio = 0.1,
    "num_train_epochs": 4,
    # Select a 2 to 10x smaller learning rate for the embedding matrices!
    "learning_rate": 5e-5,
    "embedding_learning_rate": 1e-5,
    "eval_strategy": "steps",
    "eval_steps": 10,
    "save_strategy": "epoch",
    "logging_steps": 1,
    "optim": "adamw_8bit",
    "weight_decay": 0.01,
    "lr_scheduler_type": "cosine",
    "seed": 3407,
    "output_dir": f"ckpt/pt/{args.lang}-manuals",
    "report_to": "wandb",  # Use this for WandB etc
    "save_only_model": True,  # Save only adapter weights, no optimizer state
    "max_length": max_seq_length,
}

WANDB_PROJECT = "Continued-Training-from-Manuals"
WANDB_RUN_NAME = f"{args.lang}-manuals-{datetime.now().strftime('%y%m%d-%H%M%S')}"
# Example: "r-manuals-251215-143052"

os.environ["WANDB_PROJECT"] = WANDB_PROJECT
training_args["run_name"] = WANDB_RUN_NAME

trainer = UnslothTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset["train"],
    eval_dataset = dataset["test"],
    dataset_text_field = "text",
    dataset_num_proc = 4,
    args = UnslothTrainingArguments(**training_args),
)

# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()
