# Repo-Insight

**Graph-RAG for AI Coding Agents — Complete Changes, Not Broken Code**

AI coding tools (Claude Code, Copilot, Cursor) constantly fail at one thing: when they change a function, they miss the 6 other files that depend on it. You spend 3 more prompts fixing the cascade.

Repo-Insight solves this by building a **knowledge graph** of your codebase. Before the AI writes a single line, it queries the graph to trace every caller, every dependency, every downstream effect. The result: complete change sets, not broken code.

## How It Works

```
┌──────────────┐     ┌────────────┐     ┌──────────────┐     ┌──────────┐
│  Python Repo │────▶│  parser.py │────▶│  ParsedFile  │────▶│ ingest.py│
│  (any size)  │     │ (TreeSitter)│    │  (AST data)  │     └────┬─────┘
└──────────────┘     └────────────┘     └──────────────┘          │
                                                    embedder.py ◀─┤
                                                                  │
                                                                  ▼
┌──────────────┐     ┌────────────┐     ┌──────────────┐     ┌──────────┐
│ demo_cli.py  │────▶│  agent.py  │────▶│   tools.py   │────▶│ FalkorDB │
│  (A/B demo)  │     │(ReAct loop)│     │ (7 graph ops)│     │  (graph) │
└──────────────┘     └────────────┘     └──────────────┘     └──────────┘
                          │
                          ▼
                    ┌────────────┐
                    │   SGLang   │
                    │  (Qwen3)   │
                    └────────────┘
```

## Quick Start

```bash
# 1. Start infrastructure
docker run -p 6379:6379 -p 8001:8001 falkordb/falkordb:latest
python -m sglang.launch_server --model-path Qwen/Qwen3-8B --port 30000

# 2. Install
cd Repo-Insight && python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 3. Run the demo (ingests this repo, then compares blind vs graph-grounded)
python demo_cli.py --ingest --path ./ --mode ab
```

## The A/B Demo

The CLI demonstrates the problem and the solution side-by-side:

```bash
# Ask both Mode A (blind) and Mode B (graph-grounded) to propose the same change
python demo_cli.py --ingest --path ./ --mode ab
```

**Mode A (Baseline):** The LLM proposes changes from training data. It will miss files, invent function names, and ignore dependency chains. Shown in a **red panel**.

**Mode B (Graph-Grounded):** The LLM queries the actual codebase graph *before* proposing changes. Every file reference is backed by a deterministic graph query. Shown in a **green panel** with a tool call trace.

After both modes run, a **comparison table** shows the quantitative difference:

```
┌──────────────────────────┬──────────────────┬──────────────────┐
│ Metric                   │ Mode A (Blind)   │ Mode B (Graph)   │
├──────────────────────────┼──────────────────┼──────────────────┤
│ Response Time            │ 1.2s             │ 3.8s             │
│ Graph Queries            │ 0                │ 6                │
│ Files Traced             │ 0                │ 7                │
│ Functions Found          │ 0                │ 12               │
│ Grounded in Code Graph   │ ✗ No             │ ✓ Yes            │
└──────────────────────────┴──────────────────┴──────────────────┘
```

### Demo Prompts

The CLI ships with prompts designed to expose the gap:

1. *Add a `decorators` field to the FunctionDef dataclass and update all code that creates or reads it.*
2. *Rename `get_connection` to `connect_to_graph` everywhere in the codebase.*
3. *Add error handling to `parse_file` so it returns a partial result on syntax errors.*
4. *Add a `timeout` parameter to `run_ingestion` and propagate it to all database calls.*

## Graph Tools (7 total)

| Tool | Direction | What It Does |
|------|-----------|-------------|
| `get_function_context(name)` | — | Retrieve definition, docstring, location, and module |
| `get_source_code(name)` | — | Read actual source code lines from the file |
| `get_callers(name)` | ↑ upstream | Find all functions that directly call the target |
| `get_callees(name)` | ↓ downstream | Find all functions called by the target |
| `get_blast_radius(name)` | ↑↑ transitive upstream | **What breaks if this function changes?** |
| `get_impact_radius(name)` | ↓↓ transitive downstream | What does this function touch? |
| `semantic_search(query)` | — | Vector similarity search over code embeddings |

## Running Tests

```bash
# Unit tests (no infrastructure required) — 30 tests
pytest tests/ -m "not integration"

# Integration tests (requires live FalkorDB) — 29 tests
pytest tests/ -m integration

# All tests
pytest tests/
```

## Configuration

All configuration is in `config.py`. No other module hardcodes URLs, ports, model names, or thresholds.

| Variable | Default | Description |
|----------|---------|-------------|
| `FALKORDB_HOST` | `"localhost"` | FalkorDB host address |
| `FALKORDB_PORT` | `6379` | FalkorDB port |
| `GRAPH_NAME` | `"repo_insight"` | Name of the graph in FalkorDB |
| `SGLANG_BASE_URL` | `"http://localhost:30000/v1"` | SGLang OpenAI-compatible API endpoint |
| `LLM_MODEL` | `"qwen3-8b"` | Model name served by SGLang |
| `EMBEDDING_MODEL` | `"all-MiniLM-L6-v2"` | Sentence-transformer model for embeddings |
| `IMPACT_RADIUS_MAX_DEPTH` | `2` | Max Cypher traversal depth for impact analysis |
| `AGENT_MAX_ITERATIONS` | `10` | Hard loop cap for the ReAct agent |
| `AGENT_TOOL_TIMEOUT_SECONDS` | `5` | Max seconds per individual tool call |
| `FLUSH_GRAPH_ON_INGEST` | `True` | Drop existing graph data before each ingest |

## Graph Schema

### Nodes

| Label | Properties |
|-------|------------|
| `Function` | `name`, `file_path`, `start_line`, `end_line`, `docstring`, `is_method`, `class_name`, `embedding` |
| `Class` | `name`, `file_path`, `start_line`, `end_line`, `docstring`, `embedding` |
| `Module` | `name`, `file_path` |

### Edges

| Type | From → To | Properties |
|------|-----------|------------|
| `DEFINED_IN` | `Function`/`Class` → `Module` | — |
| `IMPORTS` | `Module` → `Module` | `alias` |
| `CALLS` | `Function` → `Function` | `line`, `file_path` |

## Architecture

**Dependency chain (no circular imports):**
`parser` ← `ingest` ← `tools` ← `agent` ← `demo_cli`

**Design principles:**
- **Deterministic tools**: Every graph query returns the same result for the same graph state. No LLM involvement in tools.
- **Idempotent ingestion**: Uses `MERGE` for all node upserts. Safe to re-run.
- **File-scoped call edges**: Caller matched by `(name, file_path)` to prevent false edges across files.
- **Upstream + downstream**: `get_blast_radius` traces *who breaks* (upstream); `get_impact_radius` traces *what this touches* (downstream).

## Future: MCP Integration

The production vision is to ship `tools.py` as an **MCP (Model Context Protocol) server**. Claude Code, Copilot, and Cursor could plug into Repo-Insight natively — calling `get_blast_radius` before every edit, automatically.

```
docker-compose up  # FalkorDB + MCP server
# → Your AI coding tool now has structural awareness
```
