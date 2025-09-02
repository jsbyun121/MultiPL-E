#!/bin/bash
#
# This script launches evaluation jobs.
# Supports qwen-think, qwen-instruct, gpt-oss, or all.
# Usage:
# ./scripts/batch_run_eval.sh [model_alias] [max_tokens]
#
# Examples:
# Run only the thinking model with 4096 tokens
# ./scripts/batch_run_eval.sh qwen-think 4096
#

# --- Argument Parsing and Validation ---

model_alias=${1:-all}
max_tokens=${2:-1024}

# Validate model_alias input
if [[ "$model_alias" != "qwen-think" && "$model_alias" != "qwen-instruct" && "$model_alias" != "gpt-oss" && "$model_alias" != "all" ]]; then
    echo "Error: Invalid model alias '$model_alias'. Must be 'qwen-think', 'qwen-instruct', 'gpt-oss', or 'all'."
    exit 1
fi

# Validate max_tokens input
if ! [[ "$max_tokens" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: Max tokens must be a positive integer."
    exit 1
fi

# Languages to test
languages=("lua" "ml" "r" "rkt")

# --- Job Submission Logic ---

# Run 'qwen-instruct'
if [[ "$model_alias" == "qwen-instruct" || "$model_alias" == "all" ]]; then
    echo "--- Submitting jobs for qwen-instruct ---"
    for lang in "${languages[@]}"; do
      job_name="qwen_instruct_${lang}"
      command_to_run="bash scripts/run_eval.sh -m qwen-instruct -l ${lang}"
      echo "Submitting job: ${job_name}"
      bsr r3 1 -exclude lemon -j "${job_name}" -c "${command_to_run}"
      echo "--------------------"
    done
fi

# Run 'qwen-think'
if [[ "$model_alias" == "qwen-think" || "$model_alias" == "all" ]]; then
    echo "--- Submitting jobs for qwen-think with max_tokens=${max_tokens} ---"
    for lang in "${languages[@]}"; do
      job_name="qwen_think_${lang}_${max_tokens}"
      command_to_run="bash scripts/run_eval.sh -m qwen-think -l ${lang} -x ${max_tokens}"
      echo "Submitting job: ${job_name}"
      bsr r3 1 -exclude lemon -j "${job_name}" -c "${command_to_run}"
      echo "--------------------"
    done
fi

# Run 'gpt-oss'
if [[ "$model_alias" == "gpt-oss" || "$model_alias" == "all" ]]; then
    echo "--- Submitting jobs for gpt-oss ---"
    for lang in "${languages[@]}"; do
      job_name="gpt_oss_${lang}_${max_tokens}"
      command_to_run="bash scripts/run_eval.sh -m gpt-oss -l ${lang} -x ${max_tokens}"
      echo "Submitting job: ${job_name}"
      bsr h1 1 -j "${job_name}" -c "${command_to_run}"
      echo "--------------------"
    done
fi

echo "All requested jobs submitted."
