# Multi-Programming Language Evaluation of Large Language Models of Code (MultiPL-E)

[![Paper](https://img.shields.io/badge/paper-IEEE%20TSE-blue)](https://ieeexplore.ieee.org/abstract/document/10103177)
[![Dataset](https://img.shields.io/badge/🤗%20dataset-MultiPL--E-yellow)](https://huggingface.co/datasets/nuprl/MultiPL-E)
[![BigCode](https://img.shields.io/badge/BigCode-Evaluation%20Harness-green)](https://github.com/bigcode-project/bigcode-evaluation-harness)

## Overview

MultiPL-E is a comprehensive system for evaluating large language models on code generation tasks across multiple programming languages. It translates unit test-driven neural code generation benchmarks from Python to **18+ programming languages**, enabling robust multilingual evaluation of code LLMs.

**Key Features:**
- 🌐 **18+ Programming Languages**: From Python and JavaScript to Rust, Haskell, and OCaml
- 🔒 **Sandboxed Execution**: Safe evaluation using containerized environments
- 📊 **Statistical Analysis**: Built-in pass@k calculation and significance testing
- 🖥️ **HPC Support**: Slurm integration for large-scale evaluation
- 🤖 **Multiple Model Support**: Works with HuggingFace Transformers, OpenAI API, and custom models

## System Architecture

MultiPL-E operates on a **three-phase evaluation workflow**:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Generation    │    │    Execution     │    │    Analysis     │
│                 │    │                  │    │                 │
│ • Model loading │───▶│ • Containerized  │───▶│ • Pass@k rates  │
│ • Prompt-based  │    │   test execution │    │ • Statistical   │
│ • GPU inference │    │ • Multi-language │    │   significance  │
│ • Batch processing    │   support        │    │ • Visualization │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Repository Structure

```
MultiPL-E/
├── 📁 multipl_e/              # Core evaluation library
├── 📁 dataset_builder/        # Tools for translating benchmarks to new languages
├── 📁 datasets/               # Pre-built benchmark datasets (HumanEval, MBPP)
├── 📁 evaluation/             # Containerized execution environment
├── 📁 cluster/                # HPC/Slurm integration scripts
├── 📁 analysis/               # Statistical analysis tools (R scripts)
├── 📁 scripts/                # Utility scripts for batch operations
├── 📁 docs/                   # Jekyll documentation website
├── 📄 automodel_*.py          # Model-specific inference scripts
├── 📄 pass_k.py              # Pass@k rate calculation
└── 📄 per_problem_pass_rates.py # Detailed analysis tools
```

## Supported Languages

| Language | Extension | Status | Testing Framework |
|----------|-----------|--------|-------------------|
| Python | `.py` | ✅ Stable | pytest |
| JavaScript | `.js` | ✅ Stable | Node.js |
| TypeScript | `.ts` | ✅ Stable | TypeScript compiler |
| Java | `.java` | ✅ Stable | JUnit |
| C++ | `.cpp` | ✅ Stable | Catch2 |
| C# | `.cs` | ✅ Stable | .NET |
| Go | `.go` | ✅ Stable | go test |
| Rust | `.rs` | ✅ Stable | cargo test |
| Swift | `.swift` | ✅ Stable | XCTest |
| Ruby | `.rb` | ✅ Stable | RSpec |
| PHP | `.php` | ✅ Stable | PHPUnit |
| Scala | `.scala` | ✅ Stable | ScalaTest |
| Racket | `.rkt` | ✅ Stable | rackunit |
| Julia | `.jl` | ✅ Stable | Test.jl |
| Lua | `.lua` | ✅ Stable | LuaUnit |
| R | `.r` | ✅ Stable | testthat |
| Haskell | `.hs` | ✅ Stable | HSpec |
| OCaml | `.ml` | ✅ Stable | OUnit |
| Bash | `.sh` | ✅ Stable | Bash |
| Perl | `.pl` | ✅ Stable | Test::More |

## Prerequisites

- **Python 3.8+**
- **GPU**: CUDA-compatible GPU with 8GB+ VRAM for model inference
- **Container Runtime**: Docker or Podman for sandboxed execution
- **System Memory**: 16GB+ RAM recommended
- **Storage**: 50GB+ free space for models and datasets

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/nuprl/MultiPL-E.git
cd MultiPL-E

# 2. Install Python dependencies
pip3 install aiohttp numpy tqdm pytest datasets torch transformers

# 3. Pull the evaluation container
docker pull ghcr.io/nuprl/multipl-e-evaluation
# OR with Podman
podman pull ghcr.io/nuprl/multipl-e-evaluation
```

## Quick Start

### Basic Evaluation Example

Here's how to quickly evaluate a model on Rust with HumanEval:

```bash
# 1. Generate completions (requires GPU)
python3 automodel.py \
    --name bigcode/gpt_bigcode-santacoder \
    --root-dataset humaneval \
    --lang rs \
    --temperature 0.2 \
    --batch-size 20 \
    --completion-limit 20 \
    --output-dir-prefix results

# 2. Execute completions in sandboxed container
docker run --rm --network none \
    -v ./results:/results:rw \
    ghcr.io/nuprl/multipl-e-evaluation \
    --dir /results --output-dir /results --recursive

# 3. Calculate pass@k rates
python3 pass_k.py ./results/*
```

### Quick Start with Qwen Thinking Models

For evaluation using Qwen models with thinking capabilities:

```bash
# Evaluate Qwen model with thinking on Julia
python3 automodel_qwen_think.py \
    --name "Qwen/Qwen2.5-Coder-0.5B-Instruct" \
    --root-dataset humaneval \
    --lang jl \
    --temperature 0.2 \
    --completion-limit 20 \
    --output-dir ./outputs/qwen-julia

# Execute and analyze as above
docker run --rm --network none \
    -v ./outputs:/outputs:rw \
    ghcr.io/nuprl/multipl-e-evaluation \
    --dir /outputs/qwen-julia --output-dir /outputs/qwen-julia

python3 pass_k.py ./outputs/qwen-julia
```

**Supported Languages:** `py`, `js`, `ts`, `java`, `cpp`, `cs`, `go`, `rs`, `swift`, `rb`, `php`, `scala`, `rkt`, `jl`, `lua`, `r`, `hs`, `ml`, `sh`, `pl`

For more information:

- MultiPL-E is part of the [BigCode Code Generation LM Harness]. This
  is the easiest way to use MultiPL-E.
- The [Multilingual Code Models Evaluation] by BigCode evaluates Code LLMs
  using several benchmarks, including MultiPL-E.
- We have a [tutorial] on how to use MultiPL-E directly.
- Read our paper [MultiPL-E: A Scalable and Polyglot Approach to Benchmarking Neural Code Generation].
- The [MultiPL-E dataset] of translated prompts is available on the Hugging Face
  Hub.

## Troubleshooting

### Common Issues

#### GPU Memory Errors
```bash
# Reduce batch size for large models
python3 automodel.py --batch-size 1 ...

# Monitor GPU memory usage
nvidia-smi

# For models requiring <8GB VRAM, use smaller models
python3 automodel.py --name "microsoft/DialoGPT-medium" ...
```

#### Container Execution Failures
```bash
# Check Docker/Podman is running
docker --version
podman --version

# Verify volume mounts exist
ls -la ./results/

# Check container logs if execution fails
docker run ... --log-driver=journald

# Alternative: Use Podman instead of Docker
podman run --rm --network none -v ./results:/results:rw ...
```

#### Language-Specific Issues

**Java**: Ensure proper classpath in generated code
```bash
# Check for compilation errors in results
grep -r "compilation error" ./results/
```

**C++**: Verify compiler and headers available
```bash
# Container includes gcc/clang, but check generated includes
grep -r "#include" ./results/
```

**Swift**: Swift toolchain version compatibility
```bash
# Check Swift version in container
docker run ghcr.io/nuprl/multipl-e-evaluation swift --version
```

#### Performance Issues

**Slow Generation:**
- Use smaller models for testing: `--name "microsoft/DialoGPT-small"`
- Reduce completion limit: `--completion-limit 10`  
- Increase batch size if GPU memory allows: `--batch-size 40`

**Slow Execution:**
- Use SSD storage for container volumes
- Increase container CPU allocation: `--cpus=8`
- Process multiple languages in parallel

#### Empty Results
```bash
# Check if generation completed
ls -la ./results/*/*.json.gz

# Verify execution produced results
ls -la ./results/*/*.results.json.gz

# Check for permission issues
chmod -R 755 ./results/
```

### Getting Help

1. **Check existing issues**: [GitHub Issues](https://github.com/nuprl/MultiPL-E/issues)
2. **Discussion forum**: [GitHub Discussions](https://github.com/nuprl/MultiPL-E/discussions)
3. **BigCode community**: [Hugging Face BigCode](https://huggingface.co/bigcode)

When reporting issues, please include:
- Model name and size
- Target programming language
- Hardware specifications (GPU, RAM)
- Complete error logs
- Minimal reproduction steps

## Versions

- Version 3.0

  - We are going to maintain the changelog on the dataset page: https://huggingface.co/datasets/nuprl/MultiPL-E
  - The dataset was versioned at 3.0, and we are bumping the software version to stay in sync.
  - We have published several new PLs in the dataset. However, we have not included
    these PLs at this time: Dafny, Coq, Lean, Luau, and MATLAB.

- Version 0.5.0: Instruction-following support and new languages

  - New languages: Luau, Elixir, Lean, Coq, Dafny
  - Support for instruction-following prompts
  - vLLM support for faster evaluation

- Version 0.4.0: QoL improvements and new languages

  - New languages: OCaml, MATLAB
  - Using `.jsonl` instead of `.json` for prompts
  - Several bugfixes to prompts

- Version 0.3.0: used to evaluate [StarCoder]

  - This version corrects several bugs in prompts and test cases that resulted in lower
    pass@k rates for some of the statically typed languages. The most significant difference
    is that the pass@k for Java increases by about 2% on HumanEval.

- Version 0.2.0: used to evaluate [SantaCoder]

[tutorial]: https://nuprl.github.io/MultiPL-E/
[BigCode Code Generation LM Harness]: https://github.com/bigcode-project/bigcode-evaluation-harness
[MultiPL-E: A Scalable and Polyglot Approach to Benchmarking Neural Code Generation]: https://ieeexplore.ieee.org/abstract/document/10103177
[SantaCoder]: https://arxiv.org/abs/2301.03988
[MultiPL-E dataset]: https://huggingface.co/datasets/nuprl/MultiPL-E
[StarCoder]: https://arxiv.org/abs/2305.06161
[Multilingual Code Models Evaluation]: https://huggingface.co/spaces/bigcode/multilingual-code-evals
