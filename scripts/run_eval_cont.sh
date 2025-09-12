#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 -c <ckpt_option> -m <model_alias> -l <language> [OPTIONS]"
    echo ""
    echo "Required Arguments:"
    echo "  -c, --ckpt-option Ckpt option. Must be one of:"
    echo "                   'pt-e2'         -> Pretrained Epoch 2"
    echo "  -m, --model      Model alias. Must be one of:"
    echo "                   'qwen-think'    -> Qwen/Qwen3-4B-Thinking-2507"
    echo "                   'qwen-instruct' -> Qwen/Qwen3-4B-Instruct-2507"
    echo "                   'gpt-oss'       -> openai/gpt-oss-20b"
    echo "  -l, --lang       Programming language (r, rkt, ml, lua, jl)."
    echo ""
    echo "Optional Arguments:"
    echo "  -x, --max-tokens Maximum number of new tokens for 'qwen-think' (default: 1024)."
    echo "  -h, --help       Display this help message."
    echo ""
    echo "Example: $0 -c pt-e2 -m qwen-think -l rkt -x 4096"
    echo "Example: $0 -c pt-e2 -m qwen-instruct -l rkt"
    exit 1
}

# Default values
MODEL=""
LANG=""
MAX_TOKENS="1024" # Changed default to 1024

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
    -m|--model) MODEL="$2"; shift 2;;
    -l|--lang) LANG="$2"; shift 2;;
    -x|--max-tokens) MAX_TOKENS="$2"; shift 2;;
    -c|--ckpt-option) CKPT_OPTION="$2"; shift 2;;
    -h|--help) usage; shift 2;;
    *) echo "Error: Unknown option: $1"; usage; shift 2;;
    esac
done

# Validate required arguments
if [[ -z "$MODEL" || -z "$LANG" || -z "$CKPT_OPTION" ]]; then
    echo "Error: Model alias (-m) and language (-l) and ckpt option (-c) are required."
    usage
fi

case "$CKPT_OPTION" in
    "pt-e2")
        # Set steps per epoch for pt-e2 (epoch 2)
        case "$LANG" in
            "jl") CKPT_SUFFIX="julia-manuals/checkpoint-360" ;; 
            "lua") CKPT_SUFFIX="lua-manuals/checkpoint-56" ;;
            "ml") CKPT_SUFFIX="ocaml-manuals/checkpoint-86" ;;
            "r") CKPT_SUFFIX="r-manuals/checkpoint-198" ;;
            "rkt") CKPT_SUFFIX="racket-manuals/checkpoint-190" ;;
            *) echo "Error: Unsupported language '$LANG' for pt-e2."; usage ;;
        esac
        CKPT_DIR="continued/ckpt/pt/${CKPT_SUFFIX}" ;;
    *) echo "Error: Invalid ckpt option '$CKPT_OPTION'."; usage ;;
esac


# --- MODEL-SPECIFIC CONFIGURATION ---

# Variables for the final command
EXTRA_FLAGS=""
TOKEN_SUFFIX="" # Suffix for the output directory

case "$MODEL" in
    "qwen-think")
        MODEL_NAME="Qwen/Qwen3-4B-Thinking-2507"
        MODEL_DIR="Qwen_Qwen3-4B-Thinking-2507"
        BATCH_SIZE=16
        TEMP=0.6
        # Only 'qwen-think' uses the max-tokens flag and has a token-specific output folder
        EXTRA_FLAGS="--max-tokens ${MAX_TOKENS}"
        TOKEN_SUFFIX="_mt_${MAX_TOKENS}"
        ;;
    "qwen-instruct")
        MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507"
        MODEL_DIR="Qwen_Qwen3-4B-Instruct-2507"
        BATCH_SIZE=32
        TEMP=0.7
        TOKEN_SUFFIX="_mt_${MAX_TOKENS}"
        ;;
    "gpt-oss")
        MODEL_NAME="openai/gpt-oss-20b"
        MODEL_DIR="openai_gpt-oss-20b"
        BATCH_SIZE=8 # Use a smaller batch size for the 20B model
        TEMP=1.0
        EXTRA_FLAGS="--max-tokens ${MAX_TOKENS} --top-p 1.0"
        TOKEN_SUFFIX="_mt_${MAX_TOKENS}"
        ;;
    *) echo "Error: Invalid model alias '$MODEL'."; usage ;;
esac

# Construct a clean output directory
OUTPUT_DIR="continued/outputs/${CKPT_OPTION}/before_proc_${MODEL_DIR}${TOKEN_SUFFIX}/${LANG}"
mkdir -p ${OUTPUT_DIR}

# Construct the final command, conditionally adding flags via EXTRA_FLAGS
CMD="python automodel_instruct.py \
    --name ${MODEL_NAME} \
    --root-dataset humaneval \
    --lang ${LANG} \
    --temperature ${TEMP} \
    --completion-limit 4 \
    --output-dir ${OUTPUT_DIR} \
    --batch-size ${BATCH_SIZE} \
    --use-chat-template \
    --lora-path ${CKPT_DIR} \
    ${EXTRA_FLAGS}"

# Display and execute the command
echo "Executing command:"
# Using 'xargs' to clean up extra whitespace for cleaner output
echo "$CMD" | xargs
echo ""

python automodel_instruct.py \
    --name ${MODEL_NAME} \
    --root-dataset humaneval \
    --lang ${LANG} \
    --temperature ${TEMP} \
    --completion-limit 4 \
    --output-dir ${OUTPUT_DIR} \
    --batch-size ${BATCH_SIZE} \
    --use-chat-template \
    --lora-path ${CKPT_DIR} \
    ${EXTRA_FLAGS}
