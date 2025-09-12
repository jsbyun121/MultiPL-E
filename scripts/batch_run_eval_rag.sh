#!/bin/bash
#
# This script launches RAG evaluation jobs.
# Supports qwen-think, qwen-instruct, gpt-oss, or all.
# Usage:
# ./scripts/batch_run_eval_rag.sh [--force-choice] [model_alias] [max_tokens]
#
# Examples:
# Run with force choice enabled
# ./scripts/batch_run_eval_rag.sh --force-choice qwen-think 4096
# Run without force choice
# ./scripts/batch_run_eval_rag.sh gpt-oss 4096
# ./scripts/batch_run_eval_rag.sh qwen-instruct
#

# --- Argument Parsing and Validation ---

# Check if first argument is --force-choice
FORCE_CHOICE=false
if [[ "$1" == "--force-choice" ]]; then
    FORCE_CHOICE=true
    shift  # Remove --force-choice from arguments
fi

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
languages=("jl" "lua" "ml" "r" "rkt")

# --- Job Submission Logic ---

# Run 'qwen-instruct'
if [[ "$model_alias" == "qwen-instruct" || "$model_alias" == "all" ]]; then
    echo "--- Submitting jobs for qwen-instruct ---"
    for lang in "${languages[@]}"; do
      job_name="qwen_instruct_${lang}"
      if [[ "$FORCE_CHOICE" == true ]]; then
          command_to_run="bash scripts/run_eval_rag.sh -m qwen-instruct -l ${lang} -c"
      else
          command_to_run="bash scripts/run_eval_rag.sh -m qwen-instruct -l ${lang}"
      fi
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
      if [[ "$FORCE_CHOICE" == true ]]; then
          command_to_run="bash scripts/run_eval_rag.sh -m qwen-think -l ${lang} -x ${max_tokens} -c"
      else
          command_to_run="bash scripts/run_eval_rag.sh -m qwen-think -l ${lang} -x ${max_tokens}"
      fi
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
      if [[ "$FORCE_CHOICE" == true ]]; then
          command_to_run="bash scripts/run_eval_rag.sh -m gpt-oss -l ${lang} -x ${max_tokens} -c"
      else
          command_to_run="bash scripts/run_eval_rag.sh -m gpt-oss -l ${lang} -x ${max_tokens}"
      fi
      echo "Submitting job: ${job_name}"
      bsr a1 1 -j "${job_name}" -c "${command_to_run}"
      echo "--------------------"
    done
fi

echo "All requested jobs submitted."
