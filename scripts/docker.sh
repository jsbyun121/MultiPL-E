#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -m, --model     Model size (0.6B or 4B)"
    echo "  -t, --thinking  Thinking mode (think or nothink)"
    echo "  -l, --lang      Programming language (jl, lua, ml, r, rkt)"
    echo "  -h, --help      Display this help message"
    echo ""
    echo "Example: $0 -m 0.6B -t think -l rkt"
    exit 1
}

# Default values
MODEL=""
THINKING=""
LANG=""

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
    echo "Error: All options (model, thinking, lang) are required."
    usage
fi

# Validate model option
if [[ "$MODEL" != "0.6B" && "$MODEL" != "4B" ]]; then
    echo "Error: Model must be '0.6B' or '4B'"
    exit 1
fi

# Validate thinking option
if [[ "$THINKING" != "nothink" && "$THINKING" != "think" ]]; then
    echo "Error: Thinking mode must be 'nothink' or 'think'"
    exit 1
fi

# Validate language option
if [[ ! "$LANG" =~ ^(jl|lua|ml|r|rkt)$ ]]; then
    echo "Error: Language must be one of: jl, lua, ml, r, rkt"
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
BASE_DIR="./after_proc/${MODEL_DIR}${THINKING_SUFFIX}-4"

# Check if the base directory exists
if [[ ! -d "$BASE_DIR" ]]; then
    echo "Error: Directory $BASE_DIR does not exist"
    exit 1
fi

# Construct and execute the Docker command
CMD="docker run --rm --network none --user $(id -u):$(id -g) \
    -v ${BASE_DIR}:/out:rw \
    ghcr.io/nuprl/multipl-e-evaluation \
    --dir /out/${LANG} \
    --output-dir /out/result/${LANG}"

# Display the command that will be executed
echo "Executing Docker command:"
echo "$CMD"
echo ""

# Execute the command
eval "$CMD"