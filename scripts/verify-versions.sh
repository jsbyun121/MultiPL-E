#!/bin/bash

# verify-versions.sh - Check compiler versions in updated container
set -e

CONTAINER_IMAGE="${1:-multipl-e-updated:latest}"

echo "🔍 Verifying Compiler Versions in: $CONTAINER_IMAGE"
echo "=================================================="

docker run --rm --entrypoint="" "$CONTAINER_IMAGE" /bin/bash -c '
echo "✅ R Version:"
R --version | head -1

echo -e "\n✅ Julia Version:" 
julia --version

echo -e "\n✅ Lua Version:"
lua -v 2>&1 || lua5.4 -v 2>&1

echo -e "\n✅ Racket Version:"
racket --version

echo -e "\n✅ OCaml Version:"
ocaml -version 2>&1 || echo "OCaml not accessible (might need opam env)"
'

echo -e "\n🎯 Target Versions:"
echo "- R: 4.5.1"
echo "- Julia: 1.11.6" 
echo "- Lua: 5.4.8"
echo "- Racket: 8.18"
echo "- OCaml: 5.2.1"