# Artifact Guide

If you have questions, please contact us over email or start a discussion on the [Hugging Face Hub](https://huggingface.co/datasets/nuprl/MultiPL-T/discussions).

## Introduction

We can break the MultiPL-T artifact down into the following steps:

1. Filter Python from [The Stack] to a high-quality subset of Python
   examples, which includes LLM-generated tests that are validated by execution
   (Sections 4.1 and 4.2).
2. Use an LLM (StarCoderBase-15b) to translate the Python examples to a low-resource
   programming language. Filter out incorrect translations using test cases
   translated with MultiPL-E. We support translation to Racket, Julia, R, OCaml,
   and Lua (Section 4.3 and 4.4).
3. Fine-tune off-the-shelf LLMs on each low-resource language. (Fine-tuning
   hyperparameters are described at the top of Section 5.)
4. Evaluate the performance of these fine-tuned LLMs and compare to baselines
   (Section 5).

The paper has several other ablations and evaluations, but the steps
above describe the primary artifact.

All these steps require a GPU. Moreover, as the paper reports, doing a
complete reproduction requires:

- An estimated 550 days of aggregate datacenter GPU ([A100]) time,
- Also a significant amount of CPU time that we have not estimated, and
- Machines with 4 or 8 GPUs to fine-tune the largest models.

We have pre-built artifacts for each step of MultiPL-T and recommendations
of what is feasible to reproduce:

1. The filtered Python subset of The Stack (Step 1 above):
 
   https://huggingface.co/datasets/nuprl/stack-dedup-python-testgen-starcoder-filter-v2

   We recommend *not* attempting to rebuild this dataset. We estimate
   this required 2,000 hours on H100 / A100 GPUs and a significant amount
   of CPU time as well.

3. The MultiPL-T fine-tuning datasets for the five programming languages:

   https://huggingface.co/datasets/nuprl/MultiPL-T

   We recommend *not* attempting to rebuild this dataset. We estimate that 
   translating each language takes approximately 1,400 hours on an A100 GPU
   and a significant amount of CPU time to validate translations.

5. Fine-tuned off-the-shelf LLMs for each low-resource language. The resources
   needed to fine-tune an LLM vary significantly based on the LLM size.
   The MultiPL-T dataset is small enough that one can fine-tune StarCoderBase-1B
   in less than an hour on a consumer GPU. However, the larger models require
   several days and multi-GPU machines.

   Our fine-tuned models are available in this collection:

   https://huggingface.co/collections/nuprl/multipl-t-65242261eadae29c5faab50e

   We describe them in more detail below.

## Hardware Dependencies

### Minimum Requirements

1. A recent consumer Nvidia GPU, such as an RTX 30xx or RTX 40xx
2. At least 40GB of free disk space to install PyTorch, download LLMs, etc.
3. Linux or Windows Subsystem for Linux (WSL2). **MacOS will not work.**
4. *Recommended:* An Ampere-class (or newer) Nvidia GPU with 40GB VRAM.

### What Can Be Evaluated?

- Given a recent consumer Nvidia GPU, it is possible to re-evaluate
  StarCoderBase-1b.

- Given an Ampere-class Nvidia GPU with 20GB+ VRAM, such as an A6000 or an
  older A100, it is possible to (1) fine-tune StarCoderBase-1b and (2) evaluate
  StarCoderBase-15b. *We will attempt to provide SSH access to a 40GB A100
  for artifact evaluation.*

- On an 80GB 8xA100 node, it is possible to reproduce any part of the
  artifact. However, *the parts of the evaluation that needs 4 or 8 GPUs,
  also needs them for hours or days to complete.*

## Getting Started Guide

Please complete *Installation* and *Evaluate a Base Model* for the
kick-the-tires phase.

### Installation

It is fairly standard for SIGPLAN artifacts to be packed in a container
or VM, so that the committee does not need to bother with installation.
Unfortunately:

- Getting a GPU to work with a VM/container is extraordinarily
  complicated.
   
- The software stack that you install will depend on the GPU that you have
  available.
  
- We would need to run a container-in-a-container for evaluation, which is
  another can of worms.
  
Instead, we will guide you through installing a toolchain that works for your
hardware.

1. Basic requirements:

   a. You need to be on Linux or the Windows Subsystem for Linux (WSL2).
   
   b. You need at Python 3.10 or higher. run `python3 --version` to check your
      Python version.
   
   c. You need an Nvidia GPU with 10GB+ VRAM and CUDA 12.x (preferred) or CUDA 11.8.
      Check your VRAM and CUDA version by running `nvidia-smi`.

   d. You need Docker or Podman to run a container.

2. *Recommended:* Create and activate a virtual environment:

   ```bash
   conda create -n multiplt python=3.10
   ```

   - If activation succeeds, your CLI prompt should be prefixed with `(multiplt)`.
     **For the rest of this guide, we will assume that you're running commands
     in this virtual environment.**
   
   - Creating the environment may fail if you don't have the right dependency
     installed. If that occurs, follow the directions printed on screen to
     install the right package for your system.

3. Install PyTorch:
   
   - If you have CUDA 12.1+, you can run `pip install torch`.
   - Otherwise, see [pytorch.org](https://pytorch.org) for guidance.
   
4. Verify that PyTorch is installed and correctly detects your GPU.

   Start a Python REPL (`python`) and enter the following:

   ```python
   import torch
   torch.cuda.is_available()
   ```

   You should see `True`. Type `exit()` to quit the Python REPL.

5. Install other needed packages:

  ```bash
  python3.10 -m pip install transformers datasets accelerate
  python3.10 -m pip install 'huggingface_hub[cli]'
  ```

6. Checkout the MultiPL-E source code:

   ```bash
   git clone -b multiplt-artifact https://github.com/nuprl/MultiPL-E.git
   ```

7. Download the MultiPL-E container:

   ```bash
   docker pull ghcr.io/nuprl/multipl-e-evaluation
   ```

   (You can use `podman` instead of `docker` above.)

### Evaluate a Base Model

Before trying to evaluate a model fine-tuned with MultiPL-T, we recommend
evaluating a base model from the StarCoder or Code Llama family. Unfortunately,
to use these models, you need to create an account on huggingface.co and agree
to their terms of use. Moreover, Code Llama requires someone at Meta to
manually approve your application.

However, we have a copy of StarCoderBase-1b available that doesn't an
account. We wil walk you through evaluating this model.

1. Download the model.

  ```bash
  huggingface-cli download arjunguha/notstarcoder-1b
  ```
2. Generate completions with MultiPL-E. First, ensure you are in the MultiPL-E
   directory that you checked out earlier during Installation.
  
   ```bash
   cd MultiPL-E
   ```

   Now, generate Racket completions:

   ```bash
   python3 automodel.py --name arjunguha/notstarcoder-1b \
     --root-dataset humaneval \
     --lang rkt \
     --temperature 0.2 \
     --completion-limit 20 \
     --output-dir out \
     --batch-size 40
   ```

   This will load the model to GPU and start generating results in the `out/`
   directory. On an RTX 3080, this will take ~5m to run. A few notes and
   recommendations:

   - You can monitor GPU memory usage using `nvidia-smi`. If memory usage is
     too low, you can increase the `--batch-size`.
   - Conversely, you can decrease `--batch-size` if you get a CUDA out-of-memory
     error.
   - If you restart, MultiPL-E will not regenerate completions that are already
     saved. If you really want to regenerate completions, you can delete
     `out/*.json.gz`.

3. Execute the generated completions with MultiPL-E.

   ```bash
   docker run --rm --network none --user $(id -u):$(id -g) -v ./after_proc/qwen-0.6b-think-4:/out:rw ghcr.io/nuprl/multipl-e-evaluation --dir /out/rkt --output-dir /out/result/rkt
   ```

  A few notes:

  - This process is CPU intensive and takes about 15 minutes on a 20-core Intel
    Core i9-10900KF.
  - This command saves execution results to the `./out` directory, alongside
    the completions.
   - If you restart, MultiPL-E will not re-execute completions that it has
     already run.. If you really want to re-execute completions, you can delete
     `out/*.results.json.gz`.

4. Compute the pass rate (pass@1).

   ```bash
   python3 pass_k ./out
   ```

   You should see something like this:

   ```
   Dataset,Pass@k,Estimate,NumProblems,MinCompletions,MaxCompletions
   out,1,0.04,161,20,20
   ```

   Here is how to read it:

   - `out`: the name of the directory
   - `1`: This is pass@1, as opposed to pass@10 or pass@100
   - `0.04`: This is the pass rate (**4.4%**)
   - `161`: The number of problems evaluated. For Racket, it should be 161. It 
     is slightly lower for the other languages.
  - `20,20`: the minimum and maximum number of completions per problem. Since,
    we ran with `--num-completions 20` earlier, both should be 20. If the minimum
    is lower, either completions or executions were interrupted. You can run them
    again to continue.

5. Cross-check the pass rate with the pass rate in the paper. Table 2 lists the
   pass rate on Racket for StarCoderBase-1b as **4.7%**. We are using a
   standard, non-deterministic, sampling based LLM generation algorithm, and this
   is close enough. You can get a more stable estimate with `--num-completions 200`,
   but it will take 10x longer.

6. *Optional*. Recover some disk space.
   
   - Once you're happy with the results, you can delete the `./out` directory, 
     or rename it to something more meaningful.
  - The model consumes 5GB of disk space, and you probably want to recover it.
    To do so, run `huggingface-cli delete-cache`. You'll get a textual UI
    where you can press *space* to select the model to delete and *enter* to
    actually delete it.

Congratulations if you made it this far! Evaluating fine-tuned MultiPL-T
models is not very different from evaluating a base model.

## Quick Start with Bash Automation Script (Qwen)

For streamlined evaluation using our forked evaluation pipeline, you can use the provided automation script that simplifies the process of running evaluations across different models, thinking modes, and programming languages.

!! Execute below commands on the virtual environment!!

### Prerequisites for Bash Script

1. Ensure you have the same Python prerequisites as listed in the main tutorial
2. Make sure you have the evaluation script `automodel_qwen_think.py` in your repository
3. Ensure the script is executable: `chmod +x ./scripts/run_eval.sh`

### Using the Automation Script

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
