#!/bin/bash

# Configuration
MODELS=("0.6b" "4b")
THINKING_MODES=("think" "")
LANGUAGES=("jl" "lua" "ml" "r" "rkt")
BASE_DIR="/home/junsoo/MultiPL-E/after_proc"
RESULTS_FILE="pass_k_results_$(date +%Y%m%d_%H%M%S).txt"

# Function to log with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$RESULTS_FILE"
}

# Function to run pass_k.py and capture results
run_pass_k() {
    local model=$1
    local thinking=$2
    local lang=$3
    
    # Construct directory path
    local dir_name="qwen-${model}"
    if [[ -n "$thinking" ]]; then
        dir_name="${dir_name}-${thinking}"
    fi
    dir_name="${dir_name}-4"
    
    local result_path="${BASE_DIR}/${dir_name}/result/${lang}"
    
    # Check if directory exists
    if [[ ! -d "$result_path" ]]; then
        log_message "SKIP: Directory not found: $result_path"
        return 1
    fi
    
    # Create header for this configuration
    local config_name="Model: ${model}, Thinking: ${thinking:-none}, Language: ${lang}"
    log_message "=================================================="
    log_message "Running: $config_name"
    log_message "Path: $result_path"
    log_message "=================================================="
    
    # Run pass_k.py and capture output
    local cmd="python3 pass_k.py $result_path"
    log_message "Command: $cmd"
    
    # Execute and capture both stdout and stderr
    if output=$(python3 pass_k.py "$result_path" 2>&1); then
        echo "$output" | tee -a "$RESULTS_FILE"
        log_message "SUCCESS: Completed $config_name"
    else
        log_message "ERROR: Failed to run $config_name"
        echo "$output" | tee -a "$RESULTS_FILE"
    fi
    
    echo "" >> "$RESULTS_FILE"
}

# Main execution
main() {
    log_message "Starting pass_k evaluation for all configurations"
    log_message "Results will be saved to: $RESULTS_FILE"
    echo ""
    
    local total_configs=0
    local completed_configs=0
    local skipped_configs=0
    
    # Calculate total configurations
    total_configs=$((${#MODELS[@]} * 2 * ${#LANGUAGES[@]}))
    log_message "Total configurations to process: $total_configs"
    echo ""
    
    # Iterate through all combinations
    for model in "${MODELS[@]}"; do
        for thinking_mode in "${THINKING_MODES[@]}"; do
            for lang in "${LANGUAGES[@]}"; do
                if run_pass_k "$model" "$thinking_mode" "$lang"; then
                    ((completed_configs++))
                else
                    ((skipped_configs++))
                fi
                echo ""
            done
        done
    done
    
    # Summary
    log_message "=================================================="
    log_message "SUMMARY"
    log_message "=================================================="
    log_message "Total configurations: $total_configs"
    log_message "Completed successfully: $completed_configs"
    log_message "Skipped (directory not found): $skipped_configs"
    log_message "Results saved to: $RESULTS_FILE"
    
    if [[ $completed_configs -gt 0 ]]; then
        log_message ""
        log_message "You can view the complete results with:"
        log_message "cat $RESULTS_FILE"
    fi
}

# Check if pass_k.py exists
if [[ ! -f "pass_k.py" ]]; then
    echo "Error: pass_k.py not found in current directory"
    exit 1
fi

# Check if base directory exists
if [[ ! -d "$BASE_DIR" ]]; then
    echo "Error: Base directory $BASE_DIR not found"
    exit 1
fi

# Run main function
main