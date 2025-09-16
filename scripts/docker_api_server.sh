#!/bin/bash

# MultiPL-E API Server Docker Script
# Builds and runs the API server with all language runtimes in Docker

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Configuration
IMAGE_NAME="multipl-e-api:latest"
CONTAINER_NAME="multipl-e-api-server"
HOST_PORT="${API_PORT:-8888}"
CONTAINER_PORT="8888"

# Function to display usage
usage() {
    echo "Usage: $0 [build|run|stop|restart|logs|shell]"
    echo ""
    echo "Commands:"
    echo "  build     - Build the Docker image"
    echo "  run       - Run the API server container"
    echo "  stop      - Stop the running container"
    echo "  restart   - Stop and restart the container"
    echo "  logs      - Show container logs"
    echo "  shell     - Open shell in running container"
    echo ""
    echo "Environment Variables:"
    echo "  API_PORT  - Host port to bind to (default: 8888)"
    echo ""
    echo "Examples:"
    echo "  $0 build              # Build the image"
    echo "  $0 run                # Run on port 8888"
    echo "  API_PORT=5000 $0 run  # Run on port 5000"
    exit 1
}

build_image() {
    echo "=== Building MultiPL-E API Server Image ==="
    echo "Project directory: $PROJECT_DIR"
    echo "Image name: $IMAGE_NAME"
    echo ""

    cd "$PROJECT_DIR/evaluation"
    docker build -f Dockerfile_api -t "$IMAGE_NAME" .

    echo ""
    echo "✅ Image built successfully: $IMAGE_NAME"
}

run_container() {
    echo "=== Running MultiPL-E API Server Container ==="
    echo "Container name: $CONTAINER_NAME"
    echo "Host port: $HOST_PORT"
    echo "Container port: $CONTAINER_PORT"
    echo ""

    # Stop existing container if running
    if docker ps -q -f name="$CONTAINER_NAME" | grep -q .; then
        echo "Stopping existing container..."
        docker stop "$CONTAINER_NAME" >/dev/null
        docker rm "$CONTAINER_NAME" >/dev/null
    fi

    # Run new container
    docker run -d \
        --name "$CONTAINER_NAME" \
        -p "$HOST_PORT:$CONTAINER_PORT" \
        "$IMAGE_NAME"

    echo ""
    echo "✅ Container started successfully!"
    echo ""
    echo "API Server is running at:"
    echo "  http://localhost:$HOST_PORT"
    echo "  http://$(hostname -I | awk '{print $1}'):$HOST_PORT"
    echo ""
    echo "Endpoints:"
    echo "  GET  /health     - Health check"
    echo "  GET  /docs       - API documentation"
    echo "  POST /evaluate   - Code evaluation"
    echo ""
    echo "Check logs: $0 logs"
    echo "Stop server: $0 stop"
}

stop_container() {
    echo "=== Stopping MultiPL-E API Server ==="

    if docker ps -q -f name="$CONTAINER_NAME" | grep -q .; then
        docker stop "$CONTAINER_NAME"
        docker rm "$CONTAINER_NAME"
        echo "✅ Container stopped and removed"
    else
        echo "Container not running"
    fi
}

show_logs() {
    echo "=== MultiPL-E API Server Logs ==="
    echo "Press Ctrl+C to exit logs"
    echo ""

    if docker ps -q -f name="$CONTAINER_NAME" | grep -q .; then
        docker logs -f "$CONTAINER_NAME"
    else
        echo "Container not running"
        exit 1
    fi
}

open_shell() {
    echo "=== Opening Shell in Container ==="

    if docker ps -q -f name="$CONTAINER_NAME" | grep -q .; then
        docker exec -it "$CONTAINER_NAME" /bin/bash
    else
        echo "Container not running"
        exit 1
    fi
}

restart_container() {
    stop_container
    echo ""
    run_container
}

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in PATH"
    exit 1
fi

# Parse command
case "${1:-}" in
    build)
        build_image
        ;;
    run)
        run_container
        ;;
    stop)
        stop_container
        ;;
    restart)
        restart_container
        ;;
    logs)
        show_logs
        ;;
    shell)
        open_shell
        ;;
    *)
        usage
        ;;
esac