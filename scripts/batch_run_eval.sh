#!/bin/bash
#
# This script launches evaluation jobs.
# It allows selecting a specific model and setting max_tokens.
#
# Usage:
#   ./submit_jobs.sh [model_alias] [max_tokens]
#
# Arguments:
#   model_alias:  'qwen-think', 'qwen-instruct', or 'all' (default: 'all').
#   max_tokens:   An integer for the --max-tokens flag, used only for 'qwen-think' (default: 1024).
#
# Examples:
#   # Run only the thinking model with 4096 tokens
#   ./submit_jobs.sh qwen-think 4096
#
#   # Run only the instruct model (max_tokens argument is ignored)
#   ./submit_jobs.sh qwen-instruct
#
#   # Run both models (think model uses default 1024 tokens)
#   ./submit_jobs.sh all
#
#   # Run both models (think model uses 2048 tokens)
#   ./submit_jobs.sh all 2048

# --- Argument Parsing and Validation ---

# Set model_alias to the first argument ($1). If it's not provided, default to 'all'.
model_alias=${1:-all}
# Set max_tokens to the second argument ($2). If it's not provided, default to 1024.
max_tokens=${2:-1024}

# Validate model_alias input
if [[ "$model_alias" != "qwen-think" && "$model_alias" != "qwen-instruct" && "$model_alias" != "all" ]]; then
    echo "Error: Invalid model alias '$model_alias'. Must be 'qwen-think', 'qwen-instruct', or 'all'."
    exit 1
fi

# Validate max_tokens input
if ! [[ "$max_tokens" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: Max tokens must be a positive integer."
    exit 1
fi

# A list of all programming languages to test
languages=("jl" "lua" "ml" "r" "rkt")

# --- Job Submission Logic ---

# Run 'qwen-instruct' jobs if selected or if 'all'
if [[ "$model_alias" == "qwen-instruct" || "$model_alias" == "all" ]]; then
    echo "--- Submitting jobs for qwen-instruct ---"
    for lang in "${languages[@]}"; do
      # Note: -x flag is NOT passed here, as requested.
      job_name="qwen_instruct_${lang}"
      command_to_run="bash scripts/run_eval.sh -m qwen-instruct -l ${lang}"

      echo "Submitting job: ${job_name}"
      bsr r3 1 -j "${job_name}" -c "${command_to_run}"
      echo "--------------------"
    done
fi

# Run 'qwen-think' jobs if selected or if 'all'
if [[ "$model_alias" == "qwen-think" || "$model_alias" == "all" ]]; then
    echo "--- Submitting jobs for qwen-think with max_tokens=${max_tokens} ---"
    for lang in "${languages[@]}"; do
      # Note: -x flag IS passed here.
      job_name="qwen_think_${lang}_${max_tokens}"
      command_to_run="bash scripts/run_eval.sh -m qwen-think -l ${lang} -x ${max_tokens}"

      echo "Submitting job: ${job_name}"
      bsr r3 1 -j "${job_name}" -c "${command_to_run}"
      echo "--------------------"
    done
fi

echo "All requested jobs submitted."