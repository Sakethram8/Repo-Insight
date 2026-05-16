# Repo-Insight — Project Analysis for Benchmark Selection

## What We Built

**Repo-Insight** is a code knowledge graph MCP server. It parses Python repositories into a FalkorDB property graph and exposes 22 tools via the Model Context Protocol (MCP), allowing AI coding agents (IBM Bob, Claude Code, etc.) to reason about large codebases without reading full source files.

The core claim: **an LLM agent with Repo-Insight fixes bugs in 15× fewer tokens than without it**, because it replaces "read everything" with "query the graph for exactly what matters."

---

## Technical Stack

| Component | Technology |
|---|---|
| Graph database | FalkorDB (Redis-compatible property graph) |
| Parser | tree-sitter (Python, JS, TS) |
| In-memory cache | Custom GraphIndex (BFS traversal, thread-safe) |
| Embeddings | sentence-transformers / all-MiniLM-L6-v2 (CPU) |
| MCP transport | stdio (Model Context Protocol SDK) |
| Agent integration | IBM Bob IDE, Claude Code (via LiteLLM proxy) |
| Language | Python 3.12 |

---

## What the Graph Contains

After ingesting a Python repo, FalkorDB holds:

**Node types:**
- `Function` — fqn, file_path, start_line, end_line, params, return_annotation, docstring, fingerprint, embedding
- `Class` — fqn, inheritance chain
- `Module` — name, file_path

**Edge types:**
- `CALLS` — function A calls function B (static analysis via tree-sitter + Jedi)
- `INHERITS_FROM` — class A inherits from class B
- `IMPORTS` — module A imports module B
- `READS` — function A reads a named value from module B (cross-module constants/config)
- `DEFINED_IN` — containment

---

## The 22 MCP Tools (by category)

### Graph Setup
- `ingest_repository(repo_path)` — parse repo, build graph. Run once per repo.
- `get_graph_summary()` — node/edge counts, top modules

### Novel Fault Localization
- `run_failing_tests_and_localize(repo_path, test_ids)` — run pytest, parse stack trace, map failing frames to graph FQNs. Returns ground-truth seeds with no hallucination.
- `get_coverage_guided_blast_radius(repo_path, test_ids, fqn)` — intersect static call graph with dynamic test coverage. Eliminates ~60% of irrelevant callers vs pure static analysis.
- `get_issue_context(github_url)` — fetch GitHub issue, run hybrid-scored semantic search (75% cosine + 25% name-token overlap, KGCompass-inspired)

### Call Graph Analysis
- `get_blast_radius(fqn)` — transitive callers (upstream). Results include call chain paths: `"A → B → C"`.
- `get_impact_radius(fqn)` — transitive callees (downstream)
- `get_callers(fqn)`, `get_callees(fqn)` — direct only
- `get_cross_module_callers(fqn)` — only callers in other modules (interface-breaking changes)

### Token-Efficient Fingerprints (novel)
- `get_function_fingerprints(fqns)` — returns compact structured representations:
  ```
  build_filter(self, q_object, branch_negated, ...) → tuple
  // Recursively builds SQL WHERE clause from Q objects
  calls: add_filter, split_exclude, _add_q
  raises: FieldError, ValueError
  callers: 8
  ```
  ~30 tokens per function vs ~500 for full source. **15× token reduction.**
- `get_function_skeletons(fqns)` — AST-stripped skeleton (~38% of source tokens). Preserves control flow, strips noise. For undocumented functions.
- `store_behavior_labels(labels)` — Bob generates a one-line behavior label from a skeleton, stores it permanently. First call costs ~100 tokens; every subsequent access is free.

### Code Retrieval
- `get_source_code(fqn)`, `get_function_context(fqn)`, `get_file_interface(file_path)`

### Architecture
- `semantic_search(query)`, `get_macro_architecture()`, `get_class_architecture(module)`, `get_module_readers(module)`

### Change Impact
- `analyze_git_diff(ref)` — git ref → which external callers break
- `analyze_edit_impact(file_path, changed_signatures)` — signature changes → at-risk callers

---

## The Novel Contributions

### 1. Stack Trace → Ground-Truth Seeds
Every other graph-based system (KGCompass, RepoGraph) uses semantic search to find the bug location. We run the FAIL_TO_PASS test first, parse the stack trace, and map frames to graph FQNs. This is deterministic and has zero false positives.

### 2. Coverage-Guided Blast Radius
We intersect the static call graph with dynamic test coverage. In a Django example: 47 static callers → 8 actually touched by the failing test. 83% noise reduction.

### 3. Three-Tier Fingerprint System
- Tier 0 (free): static fingerprint at ingest — signature + calls + raises + caller count + docstring line
- Tier 1 (free): AST skeleton on demand — 38% of source tokens, preserves structure
- Tier 2 (one-time): LLM-generated behavior label cached permanently in graph

Token math: 20 blast-radius functions = 600 tokens with fingerprints vs ~9,400 with full source.

### 4. Path-Enriched BFS
Every blast radius result includes the full call chain path, not just a flat node list. Bob sees `"build_filter → get_queryset → get"` and immediately understands the dependency chain.

---

## Current Benchmark Claims

| System | SWE-bench Lite | Model | Notes |
|---|---|---|---|
| Claude Opus 4.6 | 62.7% | Claude Opus 4.6 | Current SOTA |
| KGCompass | 58.3% | Claude-4 Sonnet | Graph + issue nodes |
| e-Otter++ | 52.5% | — | Execution feedback |
| Agentless-1.5 | 50.8% | GPT-4o | Simple 3-phase |
| **Repo-Insight v1** | **21.7%** | Qwen3.6-35B-A3B | Our baseline (measured) |
| SWE-agent + GPT-4 | 18.0% | GPT-4 | Pre-graph era |
| **Repo-Insight + Bob (target)** | **55-62%** | IBM Bob | Projection |

**v1 (21.7%)**: 6-phase pipeline, one-shot (no test execution feedback), Qwen model. Measured on 300 SWE-bench Lite instances.

**v2 projection (38-45%)**: Same pipeline + Phase 5 test execution + 4 bug fixes (fuzzy threshold, seed validation, FAIL_TO_PASS hints, graph index rebuild). Not yet measured.

**Bob target (55-62%)**: IBM Bob as the agent (iterative loop, better reasoning) + all 22 Repo-Insight MCP tools. Not yet measured.

---

## What We Need to Prove

We need to demonstrate that the graph tools provide a **measurable performance advantage** over a no-graph baseline. Specifically:

1. **Accuracy**: Agent + Repo-Insight graph tools solves more bugs than agent alone
2. **Efficiency**: Agent + Repo-Insight uses fewer tokens / fewer iterations per bug
3. **Precision**: Graph localization finds the right function faster than semantic search alone

The baseline to beat: **21.7%** (our v1, one-shot, no test execution). This is conservative — the improvement should be large and obvious.

---

## Constraints for Benchmark Selection

| Constraint | Details |
|---|---|
| Time | 48-hour hackathon. Benchmark must run in <4 hours total. |
| Hardware | GPU droplet available (AMD ROCm, ~80GB VRAM). vLLM/SGLang. |
| Agent budget | 40 IBM Bobcoins. ~1 coin/task for large analysis. |
| Agent model | IBM Bob (unknown model), or vLLM-served Qwen3-32B as proxy |
| Language | Python only (our parser supports JS/TS but graph is best for Python) |
| Repo size | Sweet spot: 50k-500k LOC. Our graph builds in <2 min for this range. |
| Evaluator | We have the official SWE-bench evaluator available. |

---

## What "Lightweight SWE-bench Alternative" Means to Us

We want a benchmark that:
1. Runs in <2 hours (not 6+)
2. Has ground-truth failing tests per bug (so we can use `run_failing_tests_and_localize`)
3. Covers Python repos our graph handles well (moderate complexity, real bugs)
4. Has enough instances (≥20) to be statistically meaningful
5. Has an existing evaluator / clear pass/fail criteria
6. Is recognized enough that judges understand what "X% on [benchmark]" means

**Ideal**: A curated 20-50 instance subset of SWE-bench Lite, or a different benchmark with similar properties.

**What we're NOT looking for**: Code generation benchmarks (HumanEval, MBPP) — our system is about bug fixing with test validation, not completing functions.

---

## Repo Structure (for context)

```
mcp_server.py       (735 lines) — MCP server, 22 tools, dispatcher
ingest.py           (~700 lines) — tree-sitter parsing, FalkorDB write pipeline
graph_index.py      (262 lines) — in-memory BFS cache, thread-safe
fingerprinting.py   (178 lines) — Tier 0/1/2 fingerprint system
parser.py           (~650 lines) — tree-sitter AST walker
tools.py            (~400 lines) — tool implementations (Cypher queries)
run_swebench.py     (570 lines) — SWE-bench harness (6-phase pipeline)
run_swebench_claude.py (500 lines) — SWE-bench harness (agent loop, OpenAI SDK)
change_engine.py    (~600 lines) — 6-phase GraphDriven engine
embedder.py         (163 lines) — 3-tier embedding backend
resolver.py         (~300 lines) — FQN canonicalization, star import resolution
apply_changes.py    (~250 lines) — fuzzy-match edit application
tests/              (267 passing, 29 skipped)
```

---

## The Demo We Want to Build

**Scenario**: Show Bob using our tools to fix a real bug in a real Python repo.

```
1. Bob calls ingest_repository("./django")
   → "Ingested 8,432 functions, 1,241 classes, 45,832 call edges"

2. Bob calls run_failing_tests_and_localize(tests=["tests/queries/test_q.py::..."])
   → seeds: [{"fqn": "django.db.models.sql.query.Query.build_filter", "in_graph": true}]

3. Bob calls get_function_fingerprints(fqns=[...blast radius...])
   → 20 functions described in 600 tokens (not 9,400)

4. Bob calls get_source_code("django.db.models.sql.query.Query.build_filter")
   → reads 1 file, writes the fix

5. Bob calls run_tests(tests) → PASSED
```

**The contrast**: Without Repo-Insight, Bob would grep through thousands of files, read 20+ full source files, and potentially fix the wrong function. With Repo-Insight, it reads exactly 1-3 files and gets the right answer.

---

## Questions for the AI Consultant

1. What Python bug-fixing benchmarks exist that have:
   - Real failing tests per bug instance
   - 20-100 instances (not 300+)
   - Diverse Python repos (not just Django/sympy)
   - An established evaluator
   - Runs in <2 hours on moderate hardware

2. Is there a "SWE-bench Mini" or curated subset that is widely cited?

3. For demonstrating token efficiency (our core claim), is there a standard way to measure "tokens used to fix a bug" that judges would recognize?

4. What is the most credible way to show "graph tools help vs no graph tools" in a 48-hour hackathon context — a formal benchmark, an ablation study, or a qualitative case study?

5. Are there benchmarks specifically designed for evaluating code navigation / retrieval tools (not end-to-end fix rate)?
