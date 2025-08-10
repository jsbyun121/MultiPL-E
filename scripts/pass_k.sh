#!/bin/bash

# --- Configuration & Usage ---
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Summarizes pass@k results for the specified models."
    echo ""
    echo "Options:"
    echo "  -m, --models     A comma-separated list of model aliases to run."
    echo "                   (e.g., 'qwen-think,qwen-instruct'). Default: all."
    echo "  -x, --max-tokens The token count used for 'qwen-think' directories (default: 1024)."
    echo "  -h, --help       Display this help message."
    echo ""
    echo "Example:"
    echo "  # Summarize results for all models, assuming 1024 tokens for qwen-think"
    echo "  $0"
    echo ""
    echo "  # Summarize for qwen-think only, specifying 4096 tokens"
    echo "  $0 -m qwen-think -x 4096"
}

# --- Default Values ---
MODELS_TO_RUN="qwen-instruct,qwen-think,gpt-oss"
MAX_TOKENS="1024"
LANGUAGES=("jl" "lua" "ml" "r" "rkt")

# --- Argument Parsing ---
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--models)
            MODELS_TO_RUN="$2"
            shift 2
            ;;
        -x|--max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# --- Function to run pass_k.py ---
run_pass_k() {
    local model_alias=$1
    local lang=$2
    
    local model_dir=""
    local token_suffix=""
    
    # Determine directory structure based on model alias
    case "$model_alias" in
        "qwen-think")
            model_dir="qwen_2507_4b"
            token_suffix="_${MAX_TOKENS}"
            ;;
        "qwen-instruct")
            model_dir="qwen_2507_4b"
            ;;
        "gpt-oss")
            model_dir="gpt_oss_20b"
            ;;
        *)
            echo "ERROR: Unknown model alias '$model_alias' in run_pass_k"
            return 1
            ;;
    esac

    # Construct the path to the 'result' directory created by the Docker script
    local result_path="./after_proc_${model_dir}${token_suffix}/result/${lang}"
    
    # Check if the result directory exists
    if [[ ! -d "$result_path" ]]; then
        echo "SKIP: ${model_alias}-${lang} (directory not found at ${result_path})"
        return 1
    fi
    
    echo "Running: ${model_alias} on ${lang}"
    
    # Run pass_k.py and capture results
    if output=$(python3 pass_k.py "$result_path" 2>/dev/null); then
        # Extract pass@1 value from CSV output (3rd column)
        pass_at_1=$(echo "$output" | tail -n 1 | cut -d',' -f3)
        echo "${model_alias}-${lang}: ${pass_at_1}"
        # Append to the main results file
        echo "${model_alias}-${lang},${pass_at_1}" >> "$RESULTS_FILE"
        return 0
    else
        echo "ERROR: ${model_alias}-${lang} failed to execute pass_k.py"
        return 1
    fi
}

# --- Main Execution ---
RESULTS_FILE="results_summary_$(date +%Y%m%d_%H%M%S).csv"
echo "Results will be saved to: $RESULTS_FILE"
echo "========================================"

# Write header to results file
echo "Model-Language,Pass@1" > "$RESULTS_FILE"

total=0
completed=0

# Convert comma-separated string to array
IFS=',' read -r -a models_array <<< "$MODELS_TO_RUN"

for model in "${models_array[@]}"; do
    for lang in "${LANGUAGES[@]}"; do
        ((total++))
        if run_pass_k "$model" "$lang"; then
            ((completed++))
        fi
    done
done

# --- Summary ---
echo "========================================"
echo "Summary: $completed/$total tasks completed."
echo "Results saved to: $RESULTS_FILE"

if [[ $completed -gt 0 ]]; then
    echo ""
    echo "--- Results Summary ---"
    # Use 'column' for nice formatting if available
    if command -v column &> /dev/null; then
        cat "$RESULTS_FILE" | column -s, -t
    else
        cat "$RESULTS_FILE"
    fi
fi