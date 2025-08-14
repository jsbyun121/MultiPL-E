#!/bin/bash

# --- Usage ---
usage() {
    echo "Usage: $0 -d <result_base_dir>"
    echo ""
    echo "Summarizes pass@k results for all languages in the specified directory."
    echo ""
    echo "Options:"
    echo "  -d, --dir        Base result directory containing language subfolders."
    echo "  -h, --help       Display this help message."
    exit 1
}

# --- Parse Arguments ---
RESULT_BASE_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dir)
            RESULT_BASE_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate
if [[ -z "$RESULT_BASE_DIR" ]]; then
    echo "Error: Base result directory (-d) is required."
    usage
fi

if [[ ! -d "$RESULT_BASE_DIR" ]]; then
    echo "Error: Directory '$RESULT_BASE_DIR' does not exist."
    exit 1
fi

# --- Main Execution ---
RESULTS_FILE="results_summary_$(date +%Y%m%d_%H%M%S).csv"
echo "Results will be saved to: $RESULTS_FILE"
echo "Language,ResultPath" > "$RESULTS_FILE"

# Loop over languages
for lang_dir in "$RESULT_BASE_DIR"/*/; do
    LANG=$(basename "$lang_dir")
    echo "Processing language: $LANG"
    
    # Run pass_k.py
    if python3 pass_k.py "$lang_dir" >/dev/null 2>&1; then
        echo "$LANG,$lang_dir" >> "$RESULTS_FILE"
    else
        echo "SKIP: $lang_dir (failed)"
    fi
done

echo ""
echo "Summary: results saved to $RESULTS_FILE"
