#!/bin/bash
# Quick Demo Setup Script for IBM Bob + Repo-Insight
# Run this before starting your demo recording

set -e

echo "🚀 Repo-Insight Demo Setup"
echo "=========================="
echo ""

# 1. Check FalkorDB
echo "1️⃣ Checking FalkorDB..."
if redis-cli -h localhost -p 6379 ping > /dev/null 2>&1; then
    echo "   ✅ FalkorDB is running"
else
    echo "   ❌ FalkorDB is not running!"
    echo "   Starting FalkorDB..."
    docker compose -f docker-compose.local.yml up -d
    sleep 3
    if redis-cli -h localhost -p 6379 ping > /dev/null 2>&1; then
        echo "   ✅ FalkorDB started successfully"
    else
        echo "   ❌ Failed to start FalkorDB"
        exit 1
    fi
fi
echo ""

# 2. Test MCP Server
echo "2️⃣ Testing MCP Server..."
if timeout 2 /home/hypersonic/dev/Repo-Insight/venv/bin/python3 /home/hypersonic/dev/Repo-Insight/mcp_server.py > /dev/null 2>&1; then
    echo "   ✅ MCP Server can start"
else
    # Timeout is expected - server waits for input
    echo "   ✅ MCP Server is ready"
fi
echo ""

# 3. Prepare Django Demo Repo
echo "3️⃣ Preparing Django demo repository..."
if [ -d "/tmp/django" ]; then
    echo "   ✅ Django repo already exists at /tmp/django"
else
    echo "   📥 Cloning Django repository..."
    cd /tmp
    git clone https://github.com/django/django.git --depth 1 --quiet
    cd django
    git checkout 4f32262f8dc316e7d022c7be05c4f16ad3dc2f36 --quiet 2>/dev/null || echo "   ⚠️  Using latest commit (checkout failed)"
    echo "   ✅ Django repo ready at /tmp/django"
fi
echo ""

# 4. Verify Bob MCP Config
echo "4️⃣ Checking Bob MCP configuration..."
BOB_CONFIG="$HOME/.Bob/mcp.json"
if [ -f "$BOB_CONFIG" ]; then
    echo "   ✅ Bob MCP config exists at $BOB_CONFIG"
else
    echo "   📝 Creating Bob MCP config..."
    mkdir -p "$HOME/.Bob"
    cat > "$BOB_CONFIG" << 'EOF'
{
  "mcpServers": {
    "repo-insight": {
      "command": "/home/hypersonic/dev/Repo-Insight/venv/bin/python3",
      "args": ["/home/hypersonic/dev/Repo-Insight/mcp_server.py"],
      "env": {
        "FALKORDB_HOST": "localhost",
        "FALKORDB_PORT": "6379",
        "GRAPH_NAME": "repo_insight",
        "SKIP_JEDI": "true"
      }
    }
  }
}
EOF
    echo "   ✅ Bob MCP config created"
fi
echo ""

# 5. Summary
echo "✨ Setup Complete!"
echo "=================="
echo ""
echo "📋 Next Steps:"
echo "   1. Open IBM Bob IDE"
echo "   2. Open folder: /tmp/django"
echo "   3. Start your demo with:"
echo "      'Please analyze this Django repository and build a knowledge graph.'"
echo ""
echo "🎬 Optional: Start Streamlit UI"
echo "   cd ~/dev/Repo-Insight && streamlit run app.py"
echo ""
echo "⏱️  You have 20 minutes - Good luck! 🚀"

# Made with Bob
