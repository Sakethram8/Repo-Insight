<div align="center">
  <a href="https://git.io/typing-svg">
    <img src="https://readme-typing-svg.demolab.com?font=DM+Sans&weight=800&size=36&pause=1000&color=6366F1&center=true&width=800&height=70&lines=REPO-INSIGHT;GRAPH-DRIVEN+CODE+INTELLIGENCE;IBM+BOB+MCP+SERVER" alt="Repo-Insight">
  </a>

  <p>
    <strong>A code knowledge graph MCP server that gives IBM Bob surgical precision over any Python repository.</strong><br/>
    Bob's native tools search files with a flashlight. Repo-Insight gives Bob the map.
  </p>

  <a href="#quick-start"><img src="https://img.shields.io/badge/Quick%20Start-3%20commands-6366f1?style=for-the-badge" /></a>
  <a href="#benchmark"><img src="https://img.shields.io/badge/SWE--bench%20Lite-21.7%25%20baseline-22c55e?style=for-the-badge" /></a>
  <a href="#tools"><img src="https://img.shields.io/badge/MCP%20Tools-22-f59e0b?style=for-the-badge" /></a>
</div>

---

## What it does

Repo-Insight parses a Python repository into a **FalkorDB property graph** — every function, class, call edge, inheritance, and import relationship. It then exposes that graph as an **MCP server** that IBM Bob can call during any coding session.

When Bob needs to fix a bug, instead of searching thousands of files:

1. Bob calls `run_failing_tests_and_localize` → gets the **exact failing function** from the stack trace
2. Bob calls `get_coverage_guided_blast_radius` → gets **only the callers that the failing test actually touches**
3. Bob calls `get_issue_context` with a GitHub URL → hybrid-scored candidates with **call chain paths**
4. Bob writes the fix and verifies — guided by structure, not guesswork

**SWE-bench Lite baseline: 21.7%** (no test execution, one-shot, beating SWE-agent + GPT-4 at 18%). With Bob's iterative loop and our graph intelligence, we target **55–62%**, approaching KGCompass (58.3%, SOTA graph system).

---

## Quick Start

```bash
# 1. Start FalkorDB
docker compose -f docker-compose.local.yml up -d

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure Bob
cp .Bob/mcp.json ~/.Bob/mcp.json   # or point Bob to this repo's .Bob/mcp.json
```

Bob will now have access to all 19 Repo-Insight tools. Start by ingesting your repo:

> "Ingest the repository at /path/to/my/project"

Bob calls `ingest_repository` and the graph is ready in seconds.

---

## Bob Setup

Add `.Bob/mcp.json` to your project (already included):

```json
{
  "mcpServers": {
    "repo-insight": {
      "command": "python",
      "args": ["mcp_server.py"],
      "env": {
        "FALKORDB_HOST": "localhost",
        "FALKORDB_PORT": "6379",
        "GRAPH_NAME": "repo_insight"
      }
    }
  }
}
```

No GPU required. No external LLM API. Bob is the intelligence — Repo-Insight is the structure.

---

## Tool Reference

### Graph Setup
| Tool | Description |
|---|---|
| `ingest_repository(repo_path)` | Parse a repo and build its code graph. Call first. |
| `get_graph_summary()` | See what's currently in the graph (functions, edges, modules). |

### Dynamic Fault Localization ✦ Novel
| Tool | Description |
|---|---|
| `run_failing_tests_and_localize(repo_path, test_ids)` | Run failing tests → parse stack trace → map to graph FQNs. Deterministic, ground-truth seeds. |
| `get_coverage_guided_blast_radius(repo_path, test_ids, fqn)` | Intersect static blast radius with dynamic test coverage. Eliminates ~60% of irrelevant callers. |
| `get_issue_context(github_url)` | Fetch a GitHub issue → hybrid-scored candidate functions with call chain paths. |

### Call Graph Analysis
| Tool | Description |
|---|---|
| `get_blast_radius(fqn)` | All functions that transitively call `fqn` (upstream). What breaks if this changes? |
| `get_impact_radius(fqn)` | All functions transitively called by `fqn` (downstream). |
| `get_callers(fqn)` | Direct callers only. |
| `get_callees(fqn)` | Direct callees only. |
| `get_cross_module_callers(fqn)` | Only callers in other modules — the ones that break on interface change. |

### Code Retrieval
| Tool | Description |
|---|---|
| `get_source_code(fqn)` | Get the source of any function or class by FQN. |
| `get_function_context(fqn)` | Location, summary, module for a function. |
| `get_file_interface(file_path)` | All public function signatures in a file. |

### Architecture
| Tool | Description |
|---|---|
| `semantic_search(query)` | Find functions/classes by natural language. Hybrid cosine + name-token scoring. |
| `get_macro_architecture()` | Module-level dependency graph. |
| `get_class_architecture(module_name)` | Class-level relationships within a module. |
| `get_module_readers(module_name)` | Functions in other modules that READ values from this module. |

### Fingerprint Intelligence ✦ New
| Tool | Description |
|---|---|
| `get_function_fingerprints(fqns)` | Compact structured representation of N functions in ~600 tokens (vs ~9,400 for full source). Includes signature, behavior label, calls, raises, caller count. |
| `get_function_skeletons(fqns)` | AST-stripped code skeleton (~38% of source tokens). Preserves conditions, returns, raises — strips noise. Use to generate behavior labels cheaply. |
| `store_behavior_labels(labels)` | Cache Bob-generated behavior labels permanently in FalkorDB. First call costs ~100 tokens; every subsequent call is free. |

### Change Impact
| Tool | Description |
|---|---|
| `analyze_git_diff(ref)` | Find all callers that break due to interface changes in a git commit. |
| `analyze_edit_impact(file_path, changed_signatures)` | Given changed function signatures, find at-risk external callers. |

---

## Token Efficiency: Fingerprints vs Full Source

The fingerprint system lets Bob understand 15× more functions for the same token budget:

| Approach | Tokens | Functions understood |
|---|---|---|
| Full source (no tools) | ~9,400 | 20 |
| Fingerprints (Tier 0) | ~600 | 20 |
| Fingerprints + skeletons | ~1,360 | 20 |
| **At 2,300 tokens, Bob can cover** | **2,300** | **~80 functions** |

**How Bob uses fingerprints:**
```
1. get_blast_radius(fqn)              → 20 functions affected
2. get_function_fingerprints(fqns)    → understand all 20 in 600 tokens
3. [for undocumented complex ones]
   get_function_skeletons(fqns)       → 38%-token skeleton per function
   [Bob generates 1-line behavior label from skeleton]
   store_behavior_labels(labels)      → cached forever in FalkorDB
4. get_source_code([2-3 key fqns])   → read full source only for what matters
```

After a few sessions, 80%+ of blast-radius functions have cached labels — subsequent calls are free.

---

## Demo: Fix a Django Bug with Bob

```
User: "Fix Django issue #15234 — QuerySet filter crashes with Q objects"

Bob → get_issue_context("https://github.com/django/django/issues/15234")
   ← Top 10 candidate functions with hybrid scores + call chains

Bob → run_failing_tests_and_localize("./django", ["tests/queries/test_q.py::QTests::test_filter"])
   ← seeds: [{"fqn": "django.db.models.sql.query.Query.build_filter", "in_graph": true}]

Bob → get_coverage_guided_blast_radius("./django", ["tests/queries/test_q.py::..."], "django.db.models.sql.query.Query.build_filter")
   ← 8 affected callers (down from 47 in static blast radius)

Bob reads those 8 files, writes the fix, runs tests → resolved.
```

---

## How the Graph Works

Repo-Insight uses **tree-sitter** to parse Python AST and writes to **FalkorDB** with these edge types:

| Edge | Meaning |
|---|---|
| `CALLS` | Function A calls Function B |
| `INHERITS_FROM` | Class A inherits from Class B |
| `IMPORTS` | Module A imports Module B |
| `READS` | Function A reads a named value from Module B |
| `DEFINED_IN` | Function/Class belongs to Module |

The **GraphIndex** caches the full graph in memory for sub-millisecond BFS traversal. Every blast radius result now includes the **call chain path** (KGCompass-inspired):

```json
{
  "fqn": "django.views.generic.list.BaseListView.get",
  "distance": 2,
  "path": "build_filter → get_queryset → get"
}
```

---

## Benchmark

| System | SWE-bench Lite | Model |
|---|---|---|
| Claude Opus 4.6 | 62.7% | Claude Opus 4.6 |
| KGCompass | 58.3% | Claude-4 Sonnet |
| MiniMax M2.5 | 56.3% | MiniMax 230B |
| Agentless-1.5 | 50.8% | GPT-4o |
| **Repo-Insight v1** | **21.7%** | Qwen3.6-35BA3B |
| **Repo-Insight + Bob (target)** | **55–62%** | IBM Bob |
| SWE-agent + GPT-4 | 18.0% | GPT-4 |

Our **novel additions** vs KGCompass and RepoGraph:
- **Stack trace guided localization** — run the failing test first, use the traceback as ground-truth seeds
- **Coverage-guided blast radius** — intersect static call graph with dynamic execution coverage
- **Path-enriched responses** — every blast radius result includes the call chain, not just a node list

---

## Infrastructure

```
Developer machine / droplet
├── IBM Bob IDE           ← orchestrates everything (the agent)
│   └── calls MCP tools ──────────────────────────────────┐
├── Repo-Insight          ← MCP server (this repo)        │
│   ├── mcp_server.py     ← stdio transport               ◀┘
│   └── ingest.py         ← tree-sitter → FalkorDB
└── FalkorDB              ← Docker container, port 6379
    └── stores graph, fingerprints, embeddings (persistent)
```

No GPU required. Embeddings use `all-MiniLM-L6-v2` via sentence-transformers (CPU).

For persistent graphs: use [FalkorDB Cloud](https://app.falkordb.com) (free tier available) and set `FALKORDB_HOST` accordingly.

---

## Environment Variables

```bash
# Required
FALKORDB_HOST=localhost
FALKORDB_PORT=6379
GRAPH_NAME=repo_insight

# Optional — leave unset for local CPU embeddings
EMBED_BASE_URL=http://your-embedding-server:30001

# Optional — GitHub token for higher API rate limits in get_issue_context
GITHUB_TOKEN=ghp_...
```

Copy `.env.example` to `.env` and fill in values.
