#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -m, --model       Model size (0.6B or 4B)"
    echo "  -t, --thinking    Thinking mode (think or nothink)"
    echo "  -l, --lang        Programming language (jl, lua, ml, r, rkt)"
    echo "  -x, --max-tokens  Maximum tokens (optional, default is 2048)"
    echo "  -h, --help        Display this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -m 0.6B -t think -l rkt"
    echo "  $0 -m 4B -t nothink -l lua -x 1024"
    echo "  $0 --model 0.6B --thinking think --lang rkt --max-tokens 2048"
    exit 1
}

# Default values
MODEL=""
THINKING=""
LANG=""
MAX_TOKENS="2048"  # Set default value

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
    echo "Error: All required options (model, thinking, lang) must be specified."
    usage
fi

# Validate model option
if [[ "$MODEL" != "0.6B" && "$MODEL" != "4B" ]]; then
    echo "Error: Model must be '0.6B' or '4B'"
    echo "Provided: '$MODEL'"
    exit 1
fi

# Validate thinking option
if [[ "$THINKING" != "nothink" && "$THINKING" != "think" ]]; then
    echo "Error: Thinking mode must be 'nothink' or 'think'"
    echo "Provided: '$THINKING'"
    exit 1
fi

# Validate language option
if [[ ! "$LANG" =~ ^(jl|lua|ml|r|rkt)$ ]]; then
    echo "Error: Language must be one of: jl, lua, ml, r, rkt"
    echo "Provided: '$LANG'"
    exit 1
fi

# Validate max_tokens is a positive integer
if ! [[ "$MAX_TOKENS" =~ ^[0-9]+$ ]]; then
    echo "Error: MAX_TOKENS must be a positive integer"
    echo "Provided: '$MAX_TOKENS'"
    exit 1
fi

# Set model directory based on size
if [[ "$MODEL" == "0.6B" ]]; then
    MODEL_DIR="qwen-0.6b"
else
    MODEL_DIR="qwen-4b"
fi

# Set thinking suffix
if [[ "$THINKING" == "think" ]]; then
    THINKING_SUFFIX="-think"
else
    THINKING_SUFFIX=""
fi

# Construct the directory path
BASE_DIR="./after_proc_${MAX_TOKENS}/${MODEL_DIR}${THINKING_SUFFIX}-4"

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Thinking: $THINKING"
echo "  Language: $LANG"
echo "  Max Tokens: $MAX_TOKENS"
echo "  Base Directory: $BASE_DIR"
echo ""

# Check if the base directory exists
if [[ ! -d "$BASE_DIR" ]]; then
    echo "Error: Directory $BASE_DIR does not exist"
    echo "Please check if:"
    echo "  1. The after_proc_${MAX_TOKENS} directory exists"
    echo "  2. The model directory ${MODEL_DIR}${THINKING_SUFFIX}-4 exists within it"
    echo "  3. The path is correct relative to the current directory"
    exit 1
fi

# Check if the language subdirectory exists
if [[ ! -d "$BASE_DIR/$LANG" ]]; then
    echo "Error: Language directory $BASE_DIR/$LANG does not exist"
    exit 1
fi

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in PATH"
    exit 1
fi

# Create result directory if it doesn't exist
mkdir -p "$BASE_DIR/result/$LANG"

# Construct the Docker command
CMD="docker run --rm --network none --user $(id -u):$(id -g) \
    -v ${BASE_DIR}:/out:rw \
    ghcr.io/nuprl/multipl-e-evaluation \
    --dir /out/${LANG} \
    --output-dir /out/result/${LANG}"

# Display the command that will be executed
echo "Executing Docker command:"
echo "$CMD"
echo ""

# Execute the command and capture the exit status
if eval "$CMD"; then
    echo ""
    echo "✅ Evaluation completed successfully!"
    echo "Results saved to: $BASE_DIR/result/$LANG"
else
    echo ""
    echo "❌ Evaluation failed!"
    exit 1
fi