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

To use RAG (2nd way), install additional dependencies:
```bash
pip install langchain
pip install langchain-community
pip install langchain-experimental

pip install sentence-transformers
pip install tiktoken>=0.7.0
pip install pymupdf
pip install faiss-gpu
```

# Evaluation Methods

Supported models: Qwen3-2507, Qwen3, GPT-OSS

This repository supports two inference approaches for code generation evaluation:

## 1st Way: Direct Inference

Direct inference without retrieval augmentation. This is the standard approach for evaluating language models on code generation tasks.

### Single Language Evaluation

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

### Batch Evaluation (All Languages)

```bash
chmod +x ./scripts/batch_run_eval.sh

Example:
./scripts/batch_run_eval.sh qwen-think 4096
./scripts/batch_run_eval.sh gpt-oss 4096
./scripts/batch_run_eval.sh qwen-instruct
```

### Preprocessing

After generation, preprocess the results:

```bash
python preprocess_json.py <path/to/raw/results> <path/you/want/to/save/results> --model <model_type>

Example:
python preprocess_json.py ~/MultiPL-E/before_proc_Qwen_Qwen3-4B-Thinking-2507_mt_2048 after_proc_Qwen_Qwen3-4B-Thinking-2507_mt_2048 --model qwen-think
python preprocess_json.py ~/MultiPL-E/before_proc_Qwen_Qwen3-4B-Instruct-2507_mt_1024 after_proc_Qwen_Qwen3-4B-Instruct-2507_mt_1024 --model qwen-instruct
python preprocess_json.py ~/MultiPL-E/before_proc_openai_gpt-oss-20b_mt_4096 after_proc_openai_gpt-oss-20b_mt_4096 --model gpt-oss
```

## 2nd Way: RAG Method

Retrieval-Augmented Generation (RAG) approach that provides additional context to models during code generation.

### Single Language Evaluation

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

### Batch Evaluation (All Languages)

```bash
chmod +x ./scripts/batch_run_eval_rag.sh

Example:
./scripts/batch_run_eval_rag.sh --force-choice qwen-think 4096
./scripts/batch_run_eval_rag.sh gpt-oss 4096
./scripts/batch_run_eval_rag.sh qwen-instruct
```

### Preprocessing

After generation, preprocess the results:

```bash
python preprocess_json_rag.py <path/to/raw/results> <path/you/want/to/save/results> --model <model_type>

Example:
python preprocess_json_rag.py ~/MultiPL-E/before_proc_Qwen_Qwen3-4B-Thinking-2507_mt_2048_rag_choice after_proc_Qwen_Qwen3-4B-Thinking-2507_mt_2048_rag_choice --model qwen-think
python preprocess_json_rag.py ~/MultiPL-E/before_proc_Qwen_Qwen3-4B-Instruct-2507_mt_1024_rag_no_choice after_proc_Qwen_Qwen3-4B-Instruct-2507_mt_1024_rag_no_choice --model qwen-instruct
python preprocess_json_rag.py ~/MultiPL-E/before_proc_openai_gpt-oss-20b_mt_4096_rag_choice after_proc_openai_gpt-oss-20b_mt_4096_rag_choice --model gpt-oss
```

# Shared Evaluation Process

Both methods (direct inference and RAG) use the same evaluation and accuracy recording process:

## Docker Compilation

After preprocessing, compile results with docker:

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

## Final Evaluation

Run the final accuracy evaluation:

```bash
Think(qwen_2507_4b):
python pass_k.py ./<result_base_dir>/<lang>

Instruct(qwen_2507_4b):
python pass_k.py ./after_proc_Qwen_Qwen3-4B-Thinking-2507_mt_4096/result/<lang>
```

# REST API Server

For real-time code evaluation (e.g., RL reward systems), use the FastAPI server with all language runtimes in Docker.

## Quick Start

```bash
# Build and run API server
./scripts/docker_api_server.sh build
./scripts/docker_api_server.sh run

# Test API
python evaluation/src/test_api.py

# API Documentation
# http://localhost:8888/docs (in server node)
# http://147.46.15.142:8888/docs (in request node) (147.46.15.142 is an IP address of tao)
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check and supported languages |
| `/evaluate` | POST | Execute code and return results |
| `/docs` | GET | Interactive API documentation |

### Code Evaluation Request

```bash
curl -X POST http://147.46.15.142:8888/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "program": "result <- 2 + 2\nprint(result)",
    "language": "r"
  }'
```

### Response

```json
{
  "stdout": "[1] 4\n",
  "stderr": "",
  "exit_code": 0,
  "status": "OK",
  "execution_time_ms": 427,
  "timestamp": 1234567890
}
```

## Management

```bash
# Container management (port set as 8888)
./scripts/docker_api_server.sh stop    # Stop server
./scripts/docker_api_server.sh logs    # View logs
./scripts/docker_api_server.sh restart # Restart server

# Remote access (set environment variables)
# export API_HOST="your-server-ip" (Default: tao IP Address)
# export API_PORT="8888" (Set as same port with the node of docker container)
python evaluation/src/test_api.py
```

**Supported Languages**: Julia, OCaml, R, Racket, Lua

# Acknowledgement 

- https://github.com/nuprl/MultiPL-E
- https://github.com/nuprl/MultiPL-T