#!/bin/bash

# Quick fix for Julia and Lua issues without full rebuild

echo "🔧 Creating fixed container..."

# Create a Dockerfile that extends the current one
cat > /tmp/Dockerfile.fix << 'EOF'
FROM multipl-e-updated:latest

# Fix Julia PATH - make julia accessible to all users
RUN cp /root/.juliaup/bin/* /usr/local/bin/ 2>/dev/null || true
RUN chmod +x /usr/local/bin/julia* 2>/dev/null || true

# Fix Lua luaunit - copy existing luaunit to 5.4 path
RUN mkdir -p /usr/local/share/lua/5.4
RUN cp /usr/share/lua/5.3/luaunit.lua /usr/local/share/lua/5.4/ 2>/dev/null || true
RUN cp /usr/share/lua/5.3/luaunit.lua /usr/share/lua/5.4/ 2>/dev/null || mkdir -p /usr/share/lua/5.4 && cp /usr/share/lua/5.3/luaunit.lua /usr/share/lua/5.4/

EOF

# Build the fix
docker build -f /tmp/Dockerfile.fix -t multipl-e-updated:latest .

echo "✅ Fixed container created!"

# Test fixes
echo "🧪 Testing fixes..."
echo "Julia test:"
docker run --rm --entrypoint="" multipl-e-updated:latest /bin/bash -c "julia --version"

echo -e "\nLua test:"
docker run --rm --entrypoint="" multipl-e-updated:latest /bin/bash -c "lua -e 'require(\"luaunit\"); print(\"luaunit loaded successfully\")'"

# Cleanup
rm /tmp/Dockerfile.fix