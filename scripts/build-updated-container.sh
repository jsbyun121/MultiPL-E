#!/bin/bash

# build-updated-container.sh - Build updated MultiPL-E evaluation container

set -e

SCRIPT_DIR=$(dirname "$0")
PROJECT_DIR=$(dirname "$SCRIPT_DIR")
DOCKERFILE_PATH="$PROJECT_DIR/evaluation/Dockerfile_new"
IMAGE_NAME="multipl-e-updated:latest"

echo "🚀 Building Updated MultiPL-E Container"
echo "======================================"
echo "Dockerfile: $DOCKERFILE_PATH"
echo "Image Name: $IMAGE_NAME"
echo

# Check if Dockerfile exists
if [[ ! -f "$DOCKERFILE_PATH" ]]; then
    echo "❌ Error: Dockerfile not found at $DOCKERFILE_PATH"
    exit 1
fi

# Build the container
echo "⏳ Building container (this may take 10-15 minutes)..."
cd "$PROJECT_DIR/evaluation"

if docker build -f Dockerfile_new -t "$IMAGE_NAME" . 2>&1 | tee build.log; then
    echo "✅ Container built successfully!"
    echo
    
    # Verify versions
    echo "🔍 Verifying compiler versions..."
    bash "$SCRIPT_DIR/verify-versions.sh" "$IMAGE_NAME"
    
    echo
    echo "🎯 Next Steps:"
    echo "1. Test with small dataset: $SCRIPT_DIR/docker.sh -l r,jl,lua -d ./test-small"
    echo "2. Update docker.sh to use new image: sed -i 's|ghcr.io/nuprl/multipl-e-evaluation|$IMAGE_NAME|g' $SCRIPT_DIR/docker.sh"
    echo "3. Run full evaluation: $SCRIPT_DIR/docker.sh -l r,rkt,ml,jl,lua -d ./after_proc_Qwen_Qwen3-4B-Instruct-2507_mt_1024"
else
    echo "❌ Build failed! Check build.log for details."
    exit 1
fi