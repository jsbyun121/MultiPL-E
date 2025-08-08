#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 -m <model_alias> -l <language> [OPTIONS]"
    echo ""
    echo "Required Arguments:"
    echo "  -m, --model      Model alias. Must be one of:"
    echo "                   'qwen-think', 'qwen-instruct', 'gpt-oss'"
    echo "  -l, --lang       Programming language (e.g., rkt, lua, py)."
    echo ""
    echo "Optional Arguments:"
    echo "  -x, --max-tokens Maximum tokens used during generation (for qwen-think)."
    echo "                   This is used to find the correct input directory."
    echo "  -h, --help       Display this help message."
    echo ""
    echo "Example:"
    echo "  $0 -m qwen-think -l rkt -x 1024"
    echo "  $0 -m qwen-instruct -l py"
    exit 1
}

# Default values
MODEL=""
LANG=""
MAX_TOKENS="" # No default, only used for qwen-think

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
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate required arguments
if [[ -z "$MODEL" || -z "$LANG" ]]; then
    echo "Error: Model alias (-m) and language (-l) are required."
    usage
fi

# --- NEW LOGIC TO MATCH YOUR PYTHON SCRIPT ---

# Variables for constructing the path
MODEL_DIR=""
TOKEN_SUFFIX=""

case "$MODEL" in
    "qwen-think")
        MODEL_DIR="qwen_4b"
        if [[ -z "$MAX_TOKENS" ]]; then
            echo "Error: --max-tokens is required for the 'qwen-think' model."
            exit 1
        fi
        TOKEN_SUFFIX="_${MAX_TOKENS}"
        ;;
    "qwen-instruct")
        MODEL_DIR="qwen_4b"
        ;;
    "gpt-oss")
        MODEL_DIR="gpt_20b"
        ;;
    *)
        echo "Error: Invalid model alias '$MODEL'."
        usage
        ;;
esac

# Construct the base directory path to match the Python script's output
BASE_DIR="./after_proc_${MODEL_DIR}${TOKEN_SUFFIX}"

echo "Configuration:"
echo "  Model Alias: $MODEL"
echo "  Language: $LANG"
echo "  Target Directory: ${BASE_DIR}/${LANG}"
echo ""

# Check if the target directory exists
if [[ ! -d "${BASE_DIR}/${LANG}" ]]; then
    echo "Error: Directory ${BASE_DIR}/${LANG} does not exist."
    echo "Please ensure you have run the generation script first and the path is correct."
    exit 1
fi

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in PATH"
    exit 1
fi

# Create result directory if it doesn't exist
mkdir -p "${BASE_DIR}/result/${LANG}"

# Construct the Docker command
CMD="docker run --rm --network none --user $(id -u):$(id -g) \
    -v ${BASE_DIR}:/out:rw \
    ghcr.io/nuprl/multipl-e-evaluation \
    --dir /out/${LANG} \
    --output-dir /out/result/${LANG}"

# Display and execute the command
echo "Executing Docker command:"
echo "$CMD"
echo ""

if eval "$CMD"; then
    echo ""
    echo "✅ Evaluation completed successfully!"
    echo "Results saved to: ${BASE_DIR}/result/${LANG}"
else
    echo ""
    echo "❌ Evaluation failed!"
    exit 1
fi