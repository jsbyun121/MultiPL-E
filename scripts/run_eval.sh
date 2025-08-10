#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 -m <model_alias> -l <language> [OPTIONS]"
    echo ""
    echo "Required Arguments:"
    echo "  -m, --model      Model alias. Must be one of:"
    echo "                   'qwen-think'    -> Qwen/Qwen3-4B-Thinking-2507"
    echo "                   'qwen-instruct' -> Qwen/Qwen3-4B-Instruct-2507"
    echo "                   'gpt-oss'       -> openai/gpt-oss-20b"
    echo "  -l, --lang       Programming language (e.g., rkt, lua, py)."
    echo ""
    echo "Optional Arguments:"
    echo "  -x, --max-tokens Maximum number of new tokens for 'qwen-think' (default: 1024)."
    echo "  -r, --reasoning  Reasoning level for OpenAI model (low, medium, high). Default: 'high'."
    echo "  -h, --help       Display this help message."
    echo ""
    echo "Example: $0 -m qwen-think -l rkt -x 4096"
    echo "Example: $0 -m qwen-instruct -l py"
    exit 1
}

# Default values
MODEL=""
LANG=""
MAX_TOKENS="1024" # Changed default to 1024
REASONING="high"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
    -m|--model)
        MODEL="$2"
        shift 2
        ;;
    -l|--lang)
        LANG="$2"
        shift 2
        ;;
    -x|--max-tokens)
        MAX_TOKENS="$2"
        shift 2
        ;;
    -r|--reasoning)
        REASONING="$2"
        shift 2
        ;;
    -h|--help)
        usage
        ;;
    *)
        echo "Error: Unknown option: $1"
        usage
        ;;
    esac
done

# Validate required arguments
if [[ -z "$MODEL" || -z "$LANG" ]]; then
    echo "Error: Model alias (-m) and language (-l) are required."
    usage
fi

# --- MODEL-SPECIFIC CONFIGURATION ---

# Variables for the final command
EXTRA_FLAGS=""
TOKEN_SUFFIX="" # Suffix for the output directory

case "$MODEL" in
    "qwen-think")
        MODEL_NAME="Qwen/Qwen3-4B-Thinking-2507"
        MODEL_DIR="qwen_2507_4b"
        BATCH_SIZE=16
        TEMP=0.6
        # Only 'qwen-think' uses the max-tokens flag and has a token-specific output folder
        EXTRA_FLAGS="--max-tokens ${MAX_TOKENS}"
        TOKEN_SUFFIX="_${MAX_TOKENS}"
        ;;
    "qwen-instruct")
        MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507"
        MODEL_DIR="qwen_2507_4b"
        BATCH_SIZE=32
        TEMP=0.7
        # No extra flags or token suffix for instruct model
        ;;
    "gpt-oss")
        MODEL_NAME="openai/gpt-oss-20b"
        MODEL_DIR="gpt_oss_20b"
        BATCH_SIZE=4 # Use a smaller batch size for the 20B model
        TEMP=0.6
        EXTRA_FLAGS="--reasoning-level ${REASONING}"
        ;;
    *)
        echo "Error: Invalid model alias '$MODEL'."
        usage
        ;;
esac

# Validate reasoning level for OpenAI model
if [[ "$MODEL" == "gpt-oss" && ! "$REASONING" =~ ^(low|medium|high)$ ]]; then
    echo "Error: Reasoning level for gpt-oss must be 'low', 'medium', or 'high'."
    exit 1
fi

# Construct a clean output directory
OUTPUT_DIR="before_proc_${MODEL_DIR}${TOKEN_SUFFIX}/${LANG}"
mkdir -p ${OUTPUT_DIR}

# Construct the final command, conditionally adding flags via EXTRA_FLAGS
CMD="python automodel_qwen.py \
    --name ${MODEL_NAME} \
    --root-dataset humaneval \
    --lang ${LANG} \
    --temperature ${TEMP} \
    --completion-limit 4 \
    --output-dir ${OUTPUT_DIR} \
    --batch-size ${BATCH_SIZE} \
    --save-raw \
    --use-chat-template \
    ${EXTRA_FLAGS}"

# Display and execute the command
echo "Executing command:"
# Using 'xargs' to clean up extra whitespace for cleaner output
echo "$CMD" | xargs
echo ""
eval "$CMD" | xargs