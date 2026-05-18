# 🎬 IBM Bob + Repo-Insight Demo Cheat Sheet

**Time Limit:** 20 minutes | **Goal:** Show graph-powered code intelligence

---

## ✅ Pre-Demo Checklist (Run `./QUICK_DEMO_SETUP.sh`)

- [x] FalkorDB running on port 6379
- [x] MCP server configured at `~/.Bob/mcp.json`
- [x] Django repo cloned to `/tmp/django`
- [ ] Bob IDE ready to open

---

## 🎯 Demo Flow (5-7 minutes)

### 1. Open Bob & Ingest Repository (2 min)

**Open:** IBM Bob IDE → Open Folder → `/tmp/django`

**Prompt:**
```
Please analyze this Django repository and build a knowledge graph. 
Use the ingest_repository tool to parse all Python files.
```

**Expected:** ~6,000 functions indexed in 3-5 seconds

---

### 2. Explore with Streamlit (Optional - 2 min)

**Terminal:**
```bash
cd ~/dev/Repo-Insight && streamlit run app.py
```

**Show:** Graph visualization, semantic search, architecture diagram

---

### 3. Fix a Bug with Graph Intelligence (3 min)

**Prompt:**
```
There's a bug in Django's QuerySet.bulk_create() - it ignores batch_size. 
The test is: tests/bulk_create/tests.py::BulkCreateTests::test_batch_size

Please:
1. Run the failing test to localize the bug
2. Use get_blast_radius to understand impact
3. Fix it with minimal changes
```

**Expected Tools Used:**
- `run_failing_tests_and_localize` → Gets exact FQN
- `get_blast_radius` → Shows 12 dependent functions
- `get_source_code` → Retrieves function code
- Fix applied → Test passes

---

## 🔧 Troubleshooting

### MCP Not Connecting
```bash
# Restart Bob IDE
# Check: cat ~/.Bob/mcp.json
# Verify: /home/hypersonic/dev/Repo-Insight/venv/bin/python3 mcp_server.py
```

### FalkorDB Not Running
```bash
redis-cli -h localhost -p 6379 ping  # Should return PONG
docker compose -f docker-compose.local.yml up -d
```

### Django Repo Missing
```bash
cd /tmp && git clone https://github.com/django/django.git --depth 1
```

---

## 💡 Key Talking Points

1. **Ground Truth Localization** - Test runs FIRST, not last
2. **12.5x Token Efficiency** - Read 1 function, not 50 files
3. **Instant Blast Radius** - Know what breaks before changing
4. **3-Second Ingestion** - 6,000 functions indexed instantly

---

## 📝 Backup Prompts (If Things Go Wrong)

### If ingestion fails:
```
Can you check if the MCP server is connected? 
List the available MCP tools.
```

### If test localization fails:
```
Use semantic_search to find functions related to "bulk_create batch_size"
```

### If you run out of time:
```
Show me the architecture diagram for this repository using get_architecture_diagram
```

---

## 🎥 Recording Tips

- **Screen:** 1920x1080, full screen
- **Pace:** Slow enough to follow, fast enough to engage
- **Narration:** Confident, educational, comparative
- **Highlight:** Zoom on key moments (graph stats, tool calls, test passing)

---

## ⏱️ Time Allocation

| Phase | Time | Critical? |
|-------|------|-----------|
| Setup check | 1 min | ✅ Yes |
| Ingestion | 2 min | ✅ Yes |
| Streamlit (optional) | 2 min | ⚠️ Skip if tight |
| Bug fix demo | 3 min | ✅ Yes |
| Wrap-up | 1 min | ✅ Yes |

**Total:** 7-9 minutes (leaves buffer for issues)

---

## 🚀 Ready to Record?

1. Run: `./QUICK_DEMO_SETUP.sh`
2. Open Bob IDE
3. Open `/tmp/django`
4. Start with the ingestion prompt
5. **Breathe and have fun!** 🎬

---

**Remember:** This shows REAL usage, not perfect execution. If something breaks, explain it calmly and move forward. The value is clear even with hiccups.