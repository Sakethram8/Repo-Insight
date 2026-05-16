# Droplet Setup & Benchmark Run Guide
## Hardware: AMD MI300X · 192GB VRAM · ROCm 7.2 · 20 vCPU · 240GB RAM · 5TB scratch

---

## Hardware Context

The MI300X changes several defaults from the generic guide:

| Parameter | Generic guide | MI300X guide | Reason |
|---|---|---|---|
| Model | Qwen2.5-Coder-32B | **Qwen2.5-Coder-72B** | 192GB fits 72B in bfloat16 (~160GB) |
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
  └── vLLM port 8000                └── vLLM port 8000
      Qwen2.5-Coder-72B                 Qwen2.5-Coder-72B
HOST                              HOST
  ├── LiteLLM proxy :4000           ├── LiteLLM proxy :4000
  ├── Claude Code CLI               ├── Claude Code CLI
  ├── Repo-Insight (ibm-bob)        ├── Repo-Insight (ibm-bob)
  └── run_swebench_ccli.py          ├── FalkorDB docker :6379
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
[ROCM] python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-Coder-72B-Instruct \
  --served-model-name claude-3-5-sonnet-20241022 \
  --port 8000 \
  --host 0.0.0.0 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.90 \
  --dtype bfloat16 \
  --trust-remote-code
```

> **`--served-model-name claude-3-5-sonnet-20241022`** — Claude Code hardcodes this model
> name in every request. vLLM will serve Qwen under that alias, so Claude Code finds it
> without needing a LiteLLM proxy in the middle.

> **No `--tensor-parallel-size`** — MI300X is a single unified 192GB chip.
> If vLLM detects multiple logical GPUs and complains, add `--tensor-parallel-size 1` explicitly.

> **Model alternatives** (all fit in 192GB):
> - `Qwen/Qwen2.5-Coder-72B-Instruct` ← **recommended** (best coding, ~160GB bfloat16)
> - `Qwen/Qwen2.5-Coder-32B-Instruct` (faster inference, still very strong, ~70GB)
> - `Qwen/Qwen3-32B` (general-purpose, strong at code)

**Wait for:** `Application startup complete.` (~3-5 min to load 72B weights from disk)

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
claude --version
python -c "import mcp_server; print(len(mcp_server.TOOL_DEFINITIONS), 'MCP tools OK')"
python -c "from datasets import load_dataset; ds=load_dataset('MariusHobbhahn/swe-bench-verified-mini',split='test'); print(len(ds),'instances')"
```

---

## STEP 3 — Connect Claude Code directly to vLLM (both droplets)

vLLM natively serves the Anthropic Messages API (`/v1/messages`) — **no LiteLLM proxy needed.**
The `--served-model-name` alias in Step 1 ensures Claude Code finds the model by its expected name.

```bash
[HOST] # Point Claude Code at vLLM directly
export ANTHROPIC_BASE_URL=http://localhost:8000   # vLLM, not a proxy
export ANTHROPIC_API_KEY=fake                     # any non-empty value works
```

> If you used the container IP instead of localhost, set it here too:
> ```bash
> CONTAINER_IP=$(docker inspect rocm -f '{{.NetworkSettings.IPAddress}}')
> export ANTHROPIC_BASE_URL=http://${CONTAINER_IP}:8000
> ```

**Test full chain — Claude Code → vLLM → response:**
```bash
[HOST] claude --print -p "Reply with only the number: what is 7 times 6?"
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
export ANTHROPIC_BASE_URL=http://localhost:4000
export ANTHROPIC_API_KEY=fake
claude --print -p "List 3 Python built-in functions."
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
```bash
[HOST - DROPLET 2]
export ANTHROPIC_BASE_URL=http://localhost:4000
export ANTHROPIC_API_KEY=fake
export SKIP_JEDI=true

# Run this from the Repo-Insight directory
# Claude Code auto-reads .claude/mcp.json from the working directory
cd Repo-Insight
claude --print -p "Use the repo-insight MCP tool get_graph_summary() and tell me how many functions are in the graph."

# Expected output includes tool call and a number > 100
# If you see "get_graph_summary" in the output → MCP is working
```

---

## STEP 6 — Smoke test: 3 instances per droplet

**Run this before committing to 50 instances.**

**Droplet 1:**
```bash
[HOST - DROPLET 1]
source venv/bin/activate && cd Repo-Insight
export ANTHROPIC_BASE_URL=http://localhost:4000
export ANTHROPIC_API_KEY=fake

python run_swebench_ccli.py \
  --no-graph \
  --limit 3 \
  --workers 3 \
  --output-dir ./results/smoke_baseline

# Quick result check
python3 -c "
import json
for r in json.load(open('results/smoke_baseline/results.json')):
    print(f\"{r['instance_id'][:40]}: {r['status']} | {r['total_tokens']} tok | {r['duration_s']}s\")
"
```

**Droplet 2:**
```bash
[HOST - DROPLET 2]
source venv/bin/activate && cd Repo-Insight
export ANTHROPIC_BASE_URL=http://localhost:4000
export ANTHROPIC_API_KEY=fake
export SKIP_JEDI=true

python run_swebench_ccli.py \
  --limit 3 \
  --workers 3 \
  --output-dir ./results/smoke_graph

# Check that MCP tools were actually called
python3 -c "
import json
for r in json.load(open('results/smoke_graph/results.json')):
    print(f\"{r['instance_id'][:40]}: {r['status']} | tools={r['tools_called']}\")
"
```

**Green light criteria before full run:**
- `status` = `patched` or `empty_diff` (never `error`)
- Droplet 2: at least one instance has non-empty `tools_called`
- Duration per instance: 3-20 minutes (72B is slower than 32B per call but higher quality)
- Logs created: `ls results/smoke_*/agent_logs/`

**If something fails:**
```bash
# 1. Is the LLM responding?
curl -s http://localhost:4000/v1/models

# 2. Is FalkorDB up? (Droplet 2)
redis-cli -h localhost -p 6379 ping

# 3. Read the per-instance log
cat results/smoke_graph/agent_logs/<instance_id>.log | tail -100

# 4. Check LiteLLM log for errors
tail -50 litellm.log
```

---

## STEP 7 — Full 50-instance run

**Use tmux on each droplet** — SSH sessions will drop overnight.

```bash
[HOST] sudo apt-get install -y tmux
tmux new -s run
# Inside tmux: Ctrl+B D to detach, tmux attach -t run to re-attach
```

**Droplet 1 — baseline** (~45 minutes with 72B + 8 workers):
```bash
[HOST - DROPLET 1, in tmux]
source venv/bin/activate && cd Repo-Insight
export ANTHROPIC_BASE_URL=http://localhost:4000
export ANTHROPIC_API_KEY=fake

python run_swebench_ccli.py \
  --no-graph \
  --workers 8 \
  --output-dir ./results/ccli_baseline \
  --skip-existing
```

**Droplet 2 — graph-enhanced** (~2 hours with 72B + 8 workers):
```bash
[HOST - DROPLET 2, in tmux]
source venv/bin/activate && cd Repo-Insight
export ANTHROPIC_BASE_URL=http://localhost:4000
export ANTHROPIC_API_KEY=fake
export SKIP_JEDI=true

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
| vLLM load 72B weights | 3-5 min | 3-5 min | 0:00 |
| Host setup + install | 10 min | 10 min | 0:10 |
| LiteLLM + chain test | 5 min | 5 min | 0:15 |
| FalkorDB + ingest test | — | 5 min | 0:20 |
| Smoke test (3 instances) | 15-30 min | 20-40 min | 0:50 |
| **Full 50 instances** | **~45 min** | **~2 hours** | 2:45 |
| Comparison report | 2 min | — | 2:47 |
| Bob demo (5 instances) | — | ~1 hour | ~4:00 |
| **Total** | | | **~4 hours** |
