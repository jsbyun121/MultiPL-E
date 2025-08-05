#!/bin/bash
# Function to display usage
usage() {
echo "Usage: $0 [OPTIONS]"
echo "Options:"
echo " -m, --model Model size (0.6B or 4B)"
echo " -t, --thinking Thinking mode (nothing or think)"
echo " -l, --lang Programming language (jl, lua, ml, r, rkt)"
echo " -x, --max-tokens Maximum number of tokens (default: 1024)"
echo " -h, --help Display this help message"
echo ""
echo "Example: $0 -m 0.6B -t think -l rkt -x 2048"
exit 1
}

# Default values
MODEL=""
THINKING=""
LANG=""
MAX_TOKENS="1024"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
case $1 in
-m|--model)
MODEL="$2"
shift 2
 ;;
-t|--thinking)
THINKING="$2"
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
if [[ -z "$MODEL" || -z "$THINKING" || -z "$LANG" ]]; then
echo "Error: Model, thinking, and lang options are required."
usage
fi

# Validate model option
if [[ "$MODEL" != "0.6B" && "$MODEL" != "4B" ]]; then
echo "Error: Model must be '0.6B' or '4B'"
exit 1
fi

# Validate thinking option
if [[ "$THINKING" != "nothing" && "$THINKING" != "think" ]]; then
echo "Error: Thinking mode must be 'nothing' or 'think'"
exit 1
fi

# Validate language option
if [[ ! "$LANG" =~ ^(jl|lua|ml|r|rkt)$ ]]; then
echo "Error: Language must be one of: jl, lua, ml, r, rkt"
exit 1
fi

# Validate max-tokens option (must be positive integer)
if ! [[ "$MAX_TOKENS" =~ ^[1-9][0-9]*$ ]]; then
echo "Error: Max tokens must be a positive integer"
exit 1
fi

# Set model name based on size
if [[ "$MODEL" == "0.6B" ]]; then
MODEL_NAME="Qwen/Qwen3-0.6B"
MODEL_DIR="qwen-0.6b"
BATCH_MODEL=3
else
MODEL_NAME="Qwen/Qwen3-4B"
MODEL_DIR="qwen-4b"
BATCH_MODEL=2
fi

# Set thinking flag and directory suffix
if [[ "$THINKING" == "think" ]]; then
THINKING_FLAG=""
THINKING_SUFFIX="-think"
BATCH_THINK=12
else
THINKING_FLAG="--no-thinking"
THINKING_SUFFIX=""
BATCH_THINK=18
fi

# Construct output directory
OUTPUT_DIR="before_proc_${MAX_TOKENS}/${MODEL_DIR}${THINKING_SUFFIX}-4/${LANG}"

# Construct and execute the command
CMD="python automodel_qwen_think.py \
 --name ${MODEL_NAME} \
 --root-dataset humaneval \
 --lang ${LANG} \
 --temperature 0.7 \
 --completion-limit 4 \
 --output-dir ${OUTPUT_DIR} \
 --batch-size $((BATCH_MODEL * BATCH_THINK)) \
 --save-raw \
 --max-tokens ${MAX_TOKENS}"

# Add thinking flag if needed
if [[ -n "$THINKING_FLAG" ]]; then
CMD="$CMD $THINKING_FLAG"
fi

# Display the command that will be executed
echo "Executing command:"
echo "$CMD"
echo ""

# Execute the command
eval $CMD