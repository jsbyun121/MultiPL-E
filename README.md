# MultiPL-E (Evaluating on recent LLMs)

This repository is a fork of [MultiPL-E](https://github.com/nuprl/MultiPL-E) to evaluate recent LLMs (instruction-tuned models, thinking models, and etc.) on Low Resource Programming Languages (LRPLs).

Crucially, this fork uses a **corrected and improved version of the dataset** for five languages (**Julia, Lua, OCaml, R, Racket**) to ensure a more accurate and reliable evaluation. The original benchmark contains several errors that unfairly penalize modern models.

Detailed modifications are summarized in this [table](https://docs.google.com/spreadsheets/d/1lnDubSv39__ZuSFmnnXoXCUuPS85jcFScS9hlzI9ohI/edit?usp=sharing).

The dataset itself is also available on [Hugging Face](https://huggingface.co/datasets/jsbyun121/MultiPL-E-fixed).

# Installation

```bash
conda create -n multiplt python=3.10
pip install torch
pip install transformers datasets accelerate
pip install 'huggingface_hub[cli]'
```

To build docker container to evaluate the code generation, run
```bash
bash scripts/build-updated-container.sh
```

To enable MXFP4 inference for GPT-OSS (in Hopper or Blackwell architectures / e.g., H100 GPUs), install:
```bash
pip install triton kernels
pip install -U "git+https://github.com/triton-lang/triton.git@f33bcbd4f1051d0d9ea3fdfc0b2e68f53ededfe4#subdirectory=python/triton_kernels"
```

To use RAG, install:
```bash
pip install langchain
pip install langchain-community
pip install langchain-experimental

pip install sentence-transformers
pip install tiktoken>=0.7.0
pip install pymupdf
pip install faiss-gpu
```

## Quick Start with Bash Automation Script

Supported models: Qwen3-2507, Qwen3, GPT-OSS

The bash script provides a convenient wrapper around the evaluation pipeline with the following options:

```bash
chmod +x ./scripts/run_eval.sh
./scripts/run_eval.sh [OPTIONS]

Options:
  -m, --model         Model alias. Must be one of:
                      'qwen-think'    (-> Qwen/Qwen3-4B-Thinking-2507)
                      'qwen-instruct' (-> Qwen/Qwen3-4B-Instruct-2507)
                      'gpt-oss'       (-> openai/gpt-oss-20b)
  -l, --lang          Programming language (jl, lua, ml, r, rkt)
  -x, --max-tokens    Maximum number of tokens (default: 1024)
  -h, --help          Display help message


Example:
./scripts/run_eval.sh -m qwen-think -l rkt -x 4096

```

You can just run as below to run generations of all language at once with specific max_token_len.

```bash
chmod +x ./scripts/batch_run_eval.sh

Example:
./scripts/batch_run_eval.sh qwen-think 4096
./scripts/batch_run_eval.sh gpt-oss 4096
./scripts/batch_run_eval.sh qwen-instruct

```

After generation, you have to preprocess generations before evaluation. below is the code example.

```bash
python preprocess_json.py <path/to/raw/results> <path/you/want/to/save/results> --model <model_type>

Example:
python preprocess_json.py ~/MultiPL-E/before_proc_Qwen_Qwen3-4B-Thinking-2507_mt_2048 after_proc_Qwen_Qwen3-4B-Thinking-2507_mt_2048 --model qwen-think
python preprocess_json.py ~/MultiPL-E/before_proc_Qwen_Qwen3-4B-Instruct-2507_mt_1024 after_proc_Qwen_Qwen3-4B-Instruct-2507_mt_1024 --model qwen-instruct
python preprocess_json.py ~/MultiPL-E/before_proc_openai_gpt-oss-20b_mt_4096 after_proc_openai_gpt-oss-20b_mt_4096 --model gpt-oss

```

Beta: Currently Researching RAG method

```bash
chmod +x ./scripts/run_eval_rag.sh
./scripts/run_eval_rag.sh [OPTIONS]

Options:
  -m, --model         Model alias. Must be one of:
                      'qwen-think'    (-> Qwen/Qwen3-4B-Thinking-2507)
                      'qwen-instruct' (-> Qwen/Qwen3-4B-Instruct-2507)
                      'gpt-oss'       (-> openai/gpt-oss-20b)
  -l, --lang          Programming language (jl, lua, ml, r, rkt)
  -x, --max-tokens    Maximum number of tokens (default: 1024)
  -c, --force-choice  Force to make a query in the given list (default: False)
  -h, --help          Display help message


Example:
./scripts/run_eval_rag.sh -m qwen-think -l rkt -x 4096

```

You can just run as below to run generations of all language at once with specific max_token_len.

```bash
chmod +x ./scripts/batch_run_eval_rag.sh

Example:
./scripts/batch_run_eval_rag.sh --force-choice qwen-think 4096
./scripts/batch_run_eval_rag.sh gpt-oss 4096
./scripts/batch_run_eval_rag.sh qwen-instruct

```

After generation, you have to preprocess generations before evaluation. below is the code example.

```bash
python preprocess_json_rag.py <path/to/raw/results> <path/you/want/to/save/results> --model <model_type>

Example:
python preprocess_json_rag.py ~/MultiPL-E/before_proc_Qwen_Qwen3-4B-Thinking-2507_mt_2048_rag_choice after_proc_Qwen_Qwen3-4B-Thinking-2507_mt_2048_rag_choice --model qwen-think
python preprocess_json_rag.py ~/MultiPL-E/before_proc_Qwen_Qwen3-4B-Instruct-2507_mt_1024_rag_no_choice after_proc_Qwen_Qwen3-4B-Instruct-2507_mt_1024_rag_no_choice --model qwen-instruct
python preprocess_json_rag.py ~/MultiPL-E/before_proc_openai_gpt-oss-20b_mt_4096_rag_choice after_proc_openai_gpt-oss-20b_mt_4096_rag_choice --model gpt-oss

```

After preprocessing, you can get a compiled result json with docker.
below is the code.

```bash
chmod +x ./scripts/docker.sh
./scripts/docker.sh [Options]

Options:
  -l, --lang      Programming language (jl, lua, ml, r, rkt)
  -h, --help      Display this help message
  -d, --dir       Directory of JSON files

Example:
./scripts/docker.sh -l r,rkt,ml -d ./after_proc_Qwen_Qwen3-4B-Instruct-2507_mt_1024

```

After preprocessing, finally you can run evaluation. below is the code.

```bash
Think(qwen_2507_4b):
python pass_k.py ./<result_base_dir>/<lang>

Instruct(qwen_2507_4b):
python pass_k.py ./after_proc_Qwen_Qwen3-4B-Thinking-2507_mt_4096/result/<lang>
```


# Acknowledgement 

- https://github.com/nuprl/MultiPL-E
- https://github.com/nuprl/MultiPL-T
