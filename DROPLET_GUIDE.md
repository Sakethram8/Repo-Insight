# Droplet Setup & Benchmark Run Guide
## Hardware: AMD MI300X · 192GB VRAM · ROCm 7.2 · 20 vCPU · 240GB RAM · 5TB scratch

---

## Hardware Context

The MI300X changes several defaults from the generic guide:

| Parameter | Generic guide | MI300X guide | Reason |
|---|---|---|---|
| Model | Qwen2.5-Coder-32B | **Qwen3-Coder-30B-A3B** | MoE: 3B active, 60GB weights, fast |
| `--tensor-parallel-size` | 2 | **1** (omit flag) | MI300X is single unified 192GB chip |
| `--max-model-len` | 32768 | **65536** | 192GB headroom allows 64K context |
| Workers | 4 | **8** | 20 vCPUs supports 8 parallel agents |
| LiteLLM parallel | 8 | **16** | GPU can batch more concurrent requests |
| Expected time (baseline) | ~1.5h | **~45 min** | 8 workers |
| Expected time (graph) | ~4h | **~2h** | 8 workers + faster inference |

---

## Architecture

```
DROPLET 1 (Baseline)              DROPLET 2 (Graph-Enhanced)
─────────────────────             ─────────────────────────────
rocm container                    rocm container
  └── vLLM :8000                    └── vLLM :8000
      Qwen3-Coder-30B-A3B               Qwen3-Coder-30B-A3B
      alias: claude-3-5-sonnet          alias: claude-3-5-sonnet
HOST                              HOST
  ├── Claude Code CLI               ├── Claude Code CLI
  ├── Repo-Insight (ibm-bob)        ├── Repo-Insight (ibm-bob)
  └── run_swebench_ccli.py          ├── FalkorDB :6379
      --no-graph --workers 8        └── run_swebench_ccli.py
                                        --workers 8
```

> **Notation:**
> - `[ROCM]` — inside the rocm container: `docker exec -it rocm bash`
> - `[HOST]` — on the droplet host (normal SSH session)

---

## STEP 1 — Check / start vLLM inside ROCm container

**Enter the container:**
```bash
[HOST] docker exec -it rocm bash
```

**First, check if vLLM is already installed:**
```bash
[ROCM] python -c "import vllm; print('vLLM', vllm.__version__)"
```

**If NOT installed — install for ROCm:**
```bash
[ROCM] pip install vllm
# vLLM auto-detects ROCm if the environment is ROCm-based.
# If it installs the CUDA wheel by mistake, force the ROCm wheel:
# pip install vllm --extra-index-url https://download.pytorch.org/whl/rocm6.2
```

**Check if SGLang is already serving a model (it might be running from the original setup):**
```bash
[HOST] curl -s http://localhost:30000/v1/models 2>/dev/null | python3 -m json.tool | grep '"id"'
# If this returns a model → SGLang is running.
# You can use it directly (set api_base to :30000 in LiteLLM config) and SKIP vLLM setup.
```

**If SGLang is NOT running, start vLLM:**
```bash
[ROCM] vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --served-model-name claude-3-5-sonnet-20241022 \
  --port 8000 \
  --host 0.0.0.0 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

> **Why these flags:**
> - `--served-model-name` — aliases Qwen as `claude-3-5-sonnet-20241022` so Claude Code finds it
> - `--enable-auto-tool-choice` — required for Claude Code's `tool_choice: "auto"` requests
> - `--tool-call-parser qwen3_coder` — correct parser for Qwen3-Coder's XML tool call format
> - `--max-model-len 131072` — **128K is required**: Claude Code's system prompt + 22 MCP tool
>   definitions already fills ~57K tokens, leaving no room in a 64K window
> - No `--dtype`, no `--trust-remote-code`, no `--tensor-parallel-size` needed
> - No `--enable-reasoning` — Qwen3-**Coder** has no thinking mode (unlike Qwen3 base)
> - vLLM v0.4+ serves `/v1/messages` (Anthropic format) natively — no LiteLLM needed

> **Why Qwen3-Coder-30B-A3B:** MoE — 30B total params, only **3B active** per forward pass.
> Full weights ~60GB (trivial on 192GB), inference speed close to a 3B dense model.
> Native 256K context window; we cap at 64K for agent session length.

**Wait for:** `Application startup complete.` (~2-3 min)

**Verify from host:**
```bash
[HOST] curl -s http://localhost:8000/v1/models | python3 -m json.tool | grep '"id"'
# Expected: "id": "claude-3-5-sonnet-20241022"  ← the alias, not the real model name
```

> **If `localhost:8000` is unreachable from host:**
> ```bash
> [HOST] docker inspect rocm | grep '"IPAddress"'
> # Use the returned IP instead of localhost in all commands below
> ```

---

## STEP 2 — Host setup (run on BOTH droplets)

```bash
[HOST] # Clone the repo
git clone https://github.com/Sakethram8/Repo-Insight.git -b ibm-bob
cd Repo-Insight

# Python environment
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
pip install litellm datasets

# Claude Code CLI
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs
npm install -g @anthropic-ai/claude-code

# Verify everything
# Verify everything
claude --version
python -c "import mcp_server; print(len(mcp_server.TOOL_DEFINITIONS), 'MCP tools OK')"
python -c "from datasets import load_dataset; ds=load_dataset('MariusHobbhahn/swe-bench-verified-mini',split='test'); print(len(ds),'instances')"
```

---

## STEP 3 — Connect Claude Code directly to vLLM (both droplets)

vLLM natively serves the Anthropic Messages API (`/v1/messages`) — **no LiteLLM proxy needed.**
The `--served-model-name` alias in Step 1 ensures Claude Code finds the model by its expected name.

```bash
[HOST] export ANTHROPIC_BASE_URL=http://localhost:8000
export ANTHROPIC_API_KEY=fake
# Force Claude Code to use our model alias for ALL request tiers
# (without these it defaults to claude-opus-4-7 which vLLM won't find)
export ANTHROPIC_DEFAULT_OPUS_MODEL=claude-3-5-sonnet-20241022
export ANTHROPIC_DEFAULT_SONNET_MODEL=claude-3-5-sonnet-20241022
export ANTHROPIC_DEFAULT_HAIKU_MODEL=claude-3-5-sonnet-20241022
export CLAUDE_MODEL=claude-3-5-sonnet-20241022
```

> If the container IP differs from localhost:
> ```bash
> CONTAINER_IP=$(docker inspect rocm -f '{{.NetworkSettings.IPAddress}}')
> export ANTHROPIC_BASE_URL=http://${CONTAINER_IP}:8000
> ```

**Test full chain — Claude Code → vLLM → response:**
```bash
[HOST] claude --model claude-3-5-sonnet-20241022 --print -p "Reply with only the number: what is 7 times 6?"
# Expected: "42" — full chain confirmed
```

**If you get parameter errors** (vLLM rejecting unknown Anthropic params), fall back to LiteLLM:
```bash
# Fallback only if needed:
pip install litellm
litellm --config litellm_config.yaml --port 4000 > litellm.log 2>&1 &
export ANTHROPIC_BASE_URL=http://localhost:4000   # use proxy instead
```

---

## STEP 4 — Droplet 1 only: Baseline

No extra setup. The baseline uses Claude Code without any MCP tools.

Quick confirmation:
```bash
[HOST - DROPLET 1]
# Confirm claude can run a task (no MCP needed)
export ANTHROPIC_BASE_URL=http://localhost:8000
export ANTHROPIC_API_KEY=fake
export ANTHROPIC_DEFAULT_OPUS_MODEL=claude-3-5-sonnet-20241022
export ANTHROPIC_DEFAULT_SONNET_MODEL=claude-3-5-sonnet-20241022
export ANTHROPIC_DEFAULT_HAIKU_MODEL=claude-3-5-sonnet-20241022
export CLAUDE_MODEL=claude-3-5-sonnet-20241022
claude --model claude-3-5-sonnet-20241022 --print -p "List 3 Python built-in functions."
```

---

## STEP 5 — Droplet 2 only: Graph-enhanced setup

### 5a. Start FalkorDB
```bash
[HOST - DROPLET 2]
cd Repo-Insight
docker compose -f docker-compose.local.yml up -d
sleep 3
redis-cli -h localhost -p 6379 ping    # must return PONG
```

### 5b. Ingest Repo-Insight itself (smoke test for parser + graph)
```bash
[HOST - DROPLET 2]
source venv/bin/activate && cd Repo-Insight
export SKIP_JEDI=true

python3 -c "
from ingest import run_ingestion
r = run_ingestion('.')
print('Ingestion result:', r)
assert r.get('functions', 0) > 100, 'Too few functions — something is wrong'
print('PASS: graph built correctly')
"
```

### 5c. Verify fingerprints were written
```bash
[HOST - DROPLET 2]
python3 -c "
from ingest import get_connection
g = get_connection()
rows = g.query('MATCH (f:Function) WHERE f.fingerprint IS NOT NULL RETURN count(f)').result_set
n = rows[0][0]
print(f'Functions with fingerprints: {n}')
assert n > 100, 'Fingerprints missing — check ingest logs'
print('PASS: fingerprints OK')
"
```

### 5d. Verify Claude Code calls our MCP tools

**MCP setup for Droplet 2:**
```bash
[HOST - DROPLET 2]
# The benchmark writes config to ~/.claude.json per-instance (local scope, no approval needed).
# For manual tests, you need the user-scoped registration:
claude mcp list   # check if repo-insight is already there

# If not, add it (one-time):
claude mcp add --scope user \
  --env FALKORDB_HOST=localhost \
  --env FALKORDB_PORT=6379 \
  --env GRAPH_NAME=repo_insight \
  --env SKIP_JEDI=true \
  repo-insight -- python3 /root/Repo-Insight/mcp_server.py

claude mcp list   # confirm it appears
```

**MCP tool discovery test** (best indicator — does Claude Code actually invoke tools?)
```bash
[HOST - DROPLET 2]
export ANTHROPIC_BASE_URL=http://localhost:8000
export ANTHROPIC_API_KEY=fake
export ANTHROPIC_DEFAULT_OPUS_MODEL=claude-3-5-sonnet-20241022
export ANTHROPIC_DEFAULT_SONNET_MODEL=claude-3-5-sonnet-20241022
export ANTHROPIC_DEFAULT_HAIKU_MODEL=claude-3-5-sonnet-20241022
export CLAUDE_MODEL=claude-3-5-sonnet-20241022
export SKIP_JEDI=true

# Run from the Repo-Insight directory (where .mcp.json lives)
cd ~/Repo-Insight

# IS_SANDBOX=1 is required on root-owned droplets:
# --dangerously-skip-permissions is blocked as root without it
IS_SANDBOX=1 claude --model claude-3-5-sonnet-20241022 --print --dangerously-skip-permissions \
  -p "Call get_graph_summary to get statistics about this codebase and report the number of functions."

# Expected: structured output with functions > 100
# If you see "636 functions" or similar → full MCP chain is working ✓
```

---

## STEP 6 — Validation batch (10 instances per droplet)

Run this before the full 50-instance suite. The 10 instances are hand-picked to cover both
repos, different bug types, and different test suite sizes — enough to catch any integration
issues before committing to the full run.

**Validation batch — why these 10:**

| Instance | Repo | Bug type | FAIL tests |
|---|---|---|---|
| django__django-11790 | django | HTML attribute missing | 2 |
| django__django-11951 | django | batch_size logic | 1 |
| django__django-12193 | django | widget state bug | 1 |
| django__django-12406 | django | form/model behaviour | 3 |
| django__django-9296 | django | missing `__iter__` impl | 1 |
| sphinx-doc__sphinx-10323 | sphinx | indentation rendering | 1 |
| sphinx-doc__sphinx-7590 | sphinx | C++ literal parser | 1 |
| sphinx-doc__sphinx-8475 | sphinx | HTTP redirect handling | 1 |
| sphinx-doc__sphinx-9230 | sphinx | doc type rendering | 1 |
| sphinx-doc__sphinx-9698 | sphinx | index entry generation | 1 |

```bash
VALIDATION_IDS="django__django-11790,django__django-11951,django__django-12193,django__django-12406,django__django-9296,sphinx-doc__sphinx-10323,sphinx-doc__sphinx-7590,sphinx-doc__sphinx-8475,sphinx-doc__sphinx-9230,sphinx-doc__sphinx-9698"
```

**Droplet 1 — baseline validation:**
```bash
[HOST - DROPLET 1]
source venv/bin/activate && cd Repo-Insight
export ANTHROPIC_BASE_URL=http://localhost:8000
export ANTHROPIC_API_KEY=fake
export ANTHROPIC_DEFAULT_OPUS_MODEL=claude-3-5-sonnet-20241022
export ANTHROPIC_DEFAULT_SONNET_MODEL=claude-3-5-sonnet-20241022
export ANTHROPIC_DEFAULT_HAIKU_MODEL=claude-3-5-sonnet-20241022
export CLAUDE_MODEL=claude-3-5-sonnet-20241022
export IS_SANDBOX=1   # required: droplet runs as root
VALIDATION_IDS="django__django-11790,django__django-11951,django__django-12193,django__django-12406,django__django-9296,sphinx-doc__sphinx-10323,sphinx-doc__sphinx-7590,sphinx-doc__sphinx-8475,sphinx-doc__sphinx-9230,sphinx-doc__sphinx-9698"

python run_swebench_ccli.py \
  --no-graph \
  --instances "$VALIDATION_IDS" \
  --workers 5 \
  --output-dir ./results/validation_baseline

# Quick result check
python3 -c "
import json
from collections import Counter
rs = json.load(open('results/validation_baseline/results.json'))
print('Status breakdown:', Counter(r['status'] for r in rs))
print()
for r in rs:
    print(f\"{r['instance_id']:<45} {r['status']:<12} {r['total_tokens']:>7} tok  {r['duration_s']:>6.1f}s\")
"
```

**Droplet 2 — graph validation:**
```bash
[HOST - DROPLET 2]
source venv/bin/activate && cd Repo-Insight
export ANTHROPIC_BASE_URL=http://localhost:8000
export ANTHROPIC_API_KEY=fake
export ANTHROPIC_DEFAULT_OPUS_MODEL=claude-3-5-sonnet-20241022
export ANTHROPIC_DEFAULT_SONNET_MODEL=claude-3-5-sonnet-20241022
export ANTHROPIC_DEFAULT_HAIKU_MODEL=claude-3-5-sonnet-20241022
export CLAUDE_MODEL=claude-3-5-sonnet-20241022
export SKIP_JEDI=true
export IS_SANDBOX=1   # required: droplet runs as root
VALIDATION_IDS="django__django-11790,django__django-11951,django__django-12193,django__django-12406,django__django-9296,sphinx-doc__sphinx-10323,sphinx-doc__sphinx-7590,sphinx-doc__sphinx-8475,sphinx-doc__sphinx-9230,sphinx-doc__sphinx-9698"

python run_swebench_ccli.py \
  --instances "$VALIDATION_IDS" \
  --workers 5 \
  --output-dir ./results/validation_graph

# Check status + crucially: were MCP tools called?
python3 -c "
import json
from collections import Counter
rs = json.load(open('results/validation_graph/results.json'))
print('Status breakdown:', Counter(r['status'] for r in rs))
print()
for r in rs:
    tools = r['tools_called'][:3]  # show first 3 tools called
    print(f\"{r['instance_id']:<45} {r['status']:<12} tools={tools}\")
"
```

**Green light — proceed to full run if ALL of these are true:**

| Check | Expected | How to verify |
|---|---|---|
| No `error` statuses | All `patched` or `empty_diff` | Status breakdown above |
| Droplet 2: tools called | ≥ 5/10 instances have non-empty `tools_called` | tools= column above |
| Duration reasonable | 3-15 min/instance | `duration_s` column |
| Both repos covered | django and sphinx instances both ran | check instance IDs in output |
| Logs created | 10 log files per droplet | `ls results/validation_*/agent_logs/ \| wc -l` |

**Quick early comparison (optional but useful):**
```bash
# Run on local machine after pulling both validation results
python compare_results.py \
  --baseline results/validation_baseline/results.json \
  --graph    results/validation_graph/results.json \
  --output   results/validation_comparison.md
cat results/validation_comparison.md
# Even with 10 instances this shows whether the graph is helping
```

**If something fails:**
```bash
# 1. Is vLLM responding?
curl -s http://localhost:8000/v1/models

# 2. Is FalkorDB up? (Droplet 2)
redis-cli -h localhost -p 6379 ping

# 3. Read the specific instance log
cat results/validation_graph/agent_logs/<instance_id>.log | tail -100

# 4. Check if the model alias worked
curl -s http://localhost:8000/v1/models | grep claude
# Must show "claude-3-5-sonnet-20241022"
```

---

## STEP 7 — Full 50-instance run

**Use tmux on each droplet** — SSH sessions will drop overnight.

```bash
[HOST] sudo apt-get install -y tmux
tmux new -s run
# Inside tmux: Ctrl+B D to detach, tmux attach -t run to re-attach
```

**Droplet 1 — baseline** (~30 min with Qwen3-Coder-30B-A3B + 8 workers):
```bash
[HOST - DROPLET 1, in tmux]
source venv/bin/activate && cd Repo-Insight
export ANTHROPIC_BASE_URL=http://localhost:8000
export ANTHROPIC_API_KEY=fake
export ANTHROPIC_DEFAULT_OPUS_MODEL=claude-3-5-sonnet-20241022
export ANTHROPIC_DEFAULT_SONNET_MODEL=claude-3-5-sonnet-20241022
export ANTHROPIC_DEFAULT_HAIKU_MODEL=claude-3-5-sonnet-20241022
export CLAUDE_MODEL=claude-3-5-sonnet-20241022
export IS_SANDBOX=1   # required: droplet runs as root

# Seed output dir with validation results — skip-existing won't re-run them
cp -r results/validation_baseline/. results/ccli_baseline/ 2>/dev/null || true

python run_swebench_ccli.py \
  --no-graph \
  --workers 8 \
  --output-dir ./results/ccli_baseline \
  --skip-existing
```

**Droplet 2 — graph-enhanced** (~1.5 hours with Qwen3-Coder-30B-A3B + 8 workers):
```bash
[HOST - DROPLET 2, in tmux]
source venv/bin/activate && cd Repo-Insight
export ANTHROPIC_BASE_URL=http://localhost:8000
export ANTHROPIC_API_KEY=fake
export ANTHROPIC_DEFAULT_OPUS_MODEL=claude-3-5-sonnet-20241022
export ANTHROPIC_DEFAULT_SONNET_MODEL=claude-3-5-sonnet-20241022
export ANTHROPIC_DEFAULT_HAIKU_MODEL=claude-3-5-sonnet-20241022
export CLAUDE_MODEL=claude-3-5-sonnet-20241022
export SKIP_JEDI=true
export IS_SANDBOX=1   # required: droplet runs as root

# Seed output dir with validation results — skip-existing won't re-run them
cp -r results/validation_graph/. results/ccli_graph/ 2>/dev/null || true

python run_swebench_ccli.py \
  --workers 8 \
  --output-dir ./results/ccli_graph \
  --skip-existing
```

**Monitor from a second SSH window:**
```bash
# Patch count progress
watch -n 30 'echo "Baseline:"; ls results/ccli_baseline/*.patch 2>/dev/null | wc -l; echo "Graph:"; ls results/ccli_graph/*.patch 2>/dev/null | wc -l'

# Status breakdown
python3 -c "
import json
from collections import Counter
try:
    rs = json.load(open('results/ccli_graph/results.json'))
    print('Graph:', Counter(r['status'] for r in rs))
except: print('Graph: run in progress')
try:
    rs = json.load(open('results/ccli_baseline/results.json'))
    print('Baseline:', Counter(r['status'] for r in rs))
except: print('Baseline: run in progress')
"

# GPU utilization (inside rocm container)
docker exec rocm rocm-smi
```

---

## STEP 8 — Comparison and Bob shortlist

```bash
# Pull results to local machine (or run on Droplet 2)
scp user@droplet1:~/Repo-Insight/results/ccli_baseline/results.json ./results/baseline_results.json
scp user@droplet2:~/Repo-Insight/results/ccli_graph/results.json    ./results/graph_results.json

# Generate comparison report + Bob shortlist
python compare_results.py \
  --baseline ./results/baseline_results.json \
  --graph    ./results/graph_results.json \
  --output   ./results/comparison.md

cat results/comparison.md
```

The report ranks all 50 instances and gives you a ready-to-use Bob demo shortlist — top 5 instances where graph tools made the biggest measurable difference.

---

## STEP 9 — Bob demo (3-5 instances)

From the shortlist in `comparison.md`:

1. In Bob IDE, open Droplet 2 as workspace (FalkorDB + graphs already loaded)
2. Confirm `.Bob/mcp.json` is configured in Bob's settings
3. For each shortlisted instance, paste this into Bob:

```
Repository: [repo] cloned at [path on droplet 2]
Problem: [problem_statement from SWE-bench instance]
Failing tests: [FAIL_TO_PASS list]

Use repo-insight tools. Start with ingest_repository, then
run_failing_tests_and_localize, then get_function_fingerprints
on the blast radius. Fix the bug. Verify tests pass.
```

4. Bob calls our 23 tools, finds the bug, fixes it, runs tests
5. After all demo instances → **Export IBM Bob report**

---

## Quick reference

```bash
# Is vLLM serving? (check from host)
curl http://localhost:8000/v1/models

# Is FalkorDB up? (Droplet 2 only)
redis-cli -h localhost -p 6379 ping

# GPU utilization
docker exec rocm rocm-smi

# Live log
tail -f results/ccli_graph/logs/ccli_*.log

# Patch count progress
ls results/ccli_graph/*.patch 2>/dev/null | wc -l
ls results/ccli_baseline/*.patch 2>/dev/null | wc -l

# Status breakdown (mid-run snapshot)
python3 -c "
import json
from collections import Counter
for label, f in [('Graph','results/ccli_graph/results.json'),('Baseline','results/ccli_baseline/results.json')]:
    try: print(label, Counter(r['status'] for r in json.load(open(f))))
    except: print(label, 'still running')
"

# Read one instance's full log
cat results/ccli_graph/agent_logs/django__django-11790.log

# Run comparison at any time (mid-run is fine)
python compare_results.py \
  --baseline results/ccli_baseline/results.json \
  --graph    results/ccli_graph/results.json \
  --output   results/comparison.md
```

---

## Expected timeline (MI300X)

| Phase | Droplet 1 | Droplet 2 | Wall clock |
|---|---|---|---|
| vLLM load 30B-A3B weights | 2-3 min | 2-3 min | 0:00 |
| Host setup + install | 10 min | 10 min | 0:10 |
| LiteLLM + chain test | 5 min | 5 min | 0:15 |
| FalkorDB + ingest test | — | 5 min | 0:20 |
| Smoke test (3 instances) | 15-30 min | 20-40 min | 0:50 |
| **Full 50 instances** | **~30 min** | **~1.5 hours** | 2:00 |
| Comparison report | 2 min | — | 2:47 |
| Bob demo (5 instances) | — | ~1 hour | ~4:00 |
| **Total** | | | **~4 hours** |
