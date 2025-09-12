#!/bin/bash

# build-updated-container.sh - Build updated MultiPL-E evaluation container

set -e

SCRIPT_DIR=$(dirname "$0")
PROJECT_DIR=$(dirname "$SCRIPT_DIR")
DOCKERFILE_PATH="$PROJECT_DIR/evaluation/Dockerfile"
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

if docker build -f Dockerfile -t "$IMAGE_NAME" . 2>&1 | tee build.log; then
    echo "✅ Container built successfully!"
else
    echo "❌ Build failed! Check build.log for details."
    exit 1
fi