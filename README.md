# MultiPL-E (Evaluating on recent LLMs)

This repository is a fork of [MultiPL-E](https://github.com/nuprl/MultiPL-E) to evaluate recent LLMs (instruction-tuned models, thinking models, and etc.) on Low Resource Programming Languages (LRPLs).

# Installation

```bash
conda create -n multiplt python=3.10
pip install torch
pip install transformers datasets accelerate
pip install 'huggingface_hub[cli]'
# Download the MultiPL-E container
docker pull ghcr.io/nuprl/multipl-e-evaluation
```

## Quick Start with Bash Automation Script

Supported models: Qwen3-2507, Qwen3

The bash script provides a convenient wrapper around the evaluation pipeline with the following options:

```bash
./scripts/run_eval.sh [OPTIONS]

Options:
  -m, --model         Model size (0.6B or 4B)
  -t, --thinking      Thinking mode (nothing or think)
  -l, --lang          Programming language (jl, lua, ml, r, rkt)
  -x, --max-tokens    Maximum number of tokens (default: 1024)
  -h, --help          Display help message

Example:
./scripts/run_eval.sh -m 0.6B -t think -l rkt -x 2048

```

You can just run as below to run generations of all language at once with specific max_token_len.

```bash
chmod +x ./scripts/batch_run_eval_1024.sh
./scripts/batch_run_eval_1024.sh

chmod +x ./scripts/batch_run_eval_2048.sh
./scripts/batch_run_eval_1024.sh

chmod +x ./scripts/batch_run_eval_4096.sh
./scripts/batch_run_eval_1024.sh

```

After generation, you have to preprocess generations before evaluation. below is the code example.

```bash
python preprocess_json.py <path/to/raw/results> <path/you/want/to/save/results>

Example:
python preprocess_json.py ~/junsoo/MultiPL-E/before_proc_qwen_4b_2048 after_proc_qwen_4b_2048
python preprocess_json.py ~/junsoo/MultiPL-E/before_proc_qwen_4b after_proc_qwen_4b

```

After preprocessing, you can get a compiled result json with docker.
below is the code.

```bash
chmod +x ./scripts/docker.sh
./scripts/docker.sh [Options]

Options:
  -m, --model     Model size (0.6B or 4B)
  -t, --thinking  Thinking mode (think or nothink)
  -l, --lang      Programming language (jl, lua, ml, r, rkt)
  -h, --help      Display this help message
  -x, --max-tokens  Maximum tokens (optional, default is 2048)

Example:
./scripts/docker.sh -m 0.6B -t think -l rkt -x 2048

```

After preprocessing, finally you can run evaluation. below is the code.
```bash
chmod +x ./scripts/pass_k.sh
./scripts/pass_k.sh [Max tokens]

Example:
./scripts/pass_k.sh 1024

```

Or you can just run python file one by one.
(There is only 1 option of MAX_TOKENS for no-thinking generation which is 1024.)

```bash
Think(qwen_2507_4b):
python pass_k ./after_proc_qwen_2507_4b<_MAX_TOKENS>/result/<lang>

Instruct(qwen_2507_4b):
python pass_k ./after_proc_qwen_2507_4b/result/<lang>
```


# Acknowledgement 

- https://github.com/nuprl/MultiPL-E
- https://github.com/nuprl/MultiPL-T
