#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 -l <language1[,language2,...]> -d <base_dir>"
    echo ""
    echo "Required Arguments:"
    echo "  -l, --lang       Programming language(s), comma-separated (r, rkt, ml, lua, jl)."
    echo "  -d, --base-dir   Base directory for input/output"
    echo ""
    echo "Example:"
    echo "  $0 -l r,rkt,ml -d ./after_proc_Qwen_Qwen3-4B-Instruct-2507_mt_1024"
    exit 1
}

# Default values
LANGS=""
BASE_DIR=""

# Allowed languages
ALLOWED_LANGS=("r" "rkt" "ml" "lua" "jl")

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -l|--lang)
            LANGS="$2"
            shift 2
            ;;
        -d|--base-dir)
            BASE_DIR="$2"
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
if [[ -z "$LANGS" || -z "$BASE_DIR" ]]; then
    echo "Error: Languages (-l) and base directory (-d) are required."
    usage
fi

# Convert comma-separated list to array
IFS=',' read -r -a LANG_ARRAY <<< "$LANGS"

# Validate each language
for LANG in "${LANG_ARRAY[@]}"; do
    if [[ ! " ${ALLOWED_LANGS[*]} " =~ " ${LANG} " ]]; then
        echo "Error: Unsupported language '$LANG'. Supported languages are: r, rkt, ml, lua, jl."
        exit 1
    fi
done

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in PATH"
    exit 1
fi

# Loop over each language
for LANG in "${LANG_ARRAY[@]}"; do
    # Check if the target directory exists
    if [[ ! -d "${BASE_DIR}/${LANG}" ]]; then
        echo "Warning: Directory ${BASE_DIR}/${LANG} does not exist. Skipping."
        continue
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
    echo ""
    echo "Executing Docker command for language: $LANG"
    echo "$CMD"
    echo ""

    if eval "$CMD"; then
        echo ""
        echo "✅ Evaluation completed successfully for ${LANG}!"
        echo "Results saved to: ${BASE_DIR}/result/${LANG}"
    else
        echo ""
        echo "❌ Evaluation failed for ${LANG}!"
    fi
done
