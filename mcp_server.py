# mcp_server.py
"""
MCP (Model Context Protocol) server that exposes Repo-Insight's graph tools
as MCP tools. This allows AI coding assistants (Claude Code, Copilot, Cursor)
to natively call graph queries like get_blast_radius before every edit.

Usage:
    python mcp_server.py                    # stdio transport (default)
    python mcp_server.py --transport sse    # SSE transport for web clients

Requires:  pip install mcp
"""

import argparse
import json
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logger.warning(
        "MCP SDK not installed. Run: pip install mcp  "
        "The server will not start without it."
    )

from ingest import get_connection
from tools import (
    get_function_context, get_callers, get_callees,
    get_impact_radius, get_blast_radius, get_source_code,
    semantic_search, get_macro_architecture, get_class_architecture,
    get_cross_module_callers, get_file_interface, analyze_edit_impact,
    get_module_readers,
)
from git_tools import git_diff_impact


# ---------------------------------------------------------------------------
# Tool definitions (mirroring agent.py TOOL_SCHEMAS)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "get_function_context",
        "description": "Get the definition location, summary, and module for a function by its fully qualified name (FQN).",
        "input_schema": {
            "type": "object",
            "properties": {
                "fqn": {"type": "string", "description": "Fully Qualified Name (e.g., auth.User.login)."}
            },
            "required": ["fqn"],
        },
    },
    {
        "name": "get_callers",
        "description": "Find all functions that directly call the specified function.",
        "input_schema": {
            "type": "object",
            "properties": {
                "fqn": {"type": "string", "description": "FQN of the function to find callers of."}
            },
            "required": ["fqn"],
        },
    },
    {
        "name": "get_callees",
        "description": "Find all functions directly called by the specified function.",
        "input_schema": {
            "type": "object",
            "properties": {
                "fqn": {"type": "string", "description": "FQN of the function to find callees of."}
            },
            "required": ["fqn"],
        },
    },
    {
        "name": "get_impact_radius",
        "description": "Find all functions transitively called BY the function (downstream). Shows what this function touches.",
        "input_schema": {
            "type": "object",
            "properties": {
                "fqn": {"type": "string", "description": "FQN of the function to trace downstream from."}
            },
            "required": ["fqn"],
        },
    },
    {
        "name": "get_blast_radius",
        "description": "Find all functions that transitively CALL the function (upstream). Shows what breaks if this function changes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "fqn": {"type": "string", "description": "FQN of the function to assess breakage risk for."}
            },
            "required": ["fqn"],
        },
    },
    {
        "name": "get_source_code",
        "description": "Retrieve the actual source code of a function or class by FQN.",
        "input_schema": {
            "type": "object",
            "properties": {
                "fqn": {"type": "string", "description": "FQN of the function/class to retrieve source code for."}
            },
            "required": ["fqn"],
        },
    },
    {
        "name": "semantic_search",
        "description": "Find functions or classes semantically similar to a natural language query.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language description of what to find."}
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_macro_architecture",
        "description": "Get the high-level macro architecture (Thick Edges) between modules based on calls, imports, and inheritance.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_class_architecture",
        "description": "Get the medium-level architecture within a specific module based on calls and inheritance.",
        "input_schema": {
            "type": "object",
            "properties": {
                "module_name": {"type": "string", "description": "Name of the module (e.g., auth.models)."}
            },
            "required": ["module_name"],
        },
    },
    {
        "name": "get_module_readers",
        "description": (
            "Find functions in other modules that READ (not call) named values exported from a module — "
            "constants, config values, class instances, type aliases. "
            "These functions depend on the VALUE of those names, not just their call signature. "
            "If you change DATABASE_URL in config.py, all functions that read it from other modules may break. "
            "Use this alongside get_cross_module_callers for complete impact coverage."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "module_name": {"type": "string", "description": "Dotted module name (e.g., config, mypackage.settings)."}
            },
            "required": ["module_name"],
        },
    },
    {
        "name": "get_cross_module_callers",
        "description": (
            "Find all callers of a function that live in a different module. "
            "These are the only callers that could break if the function's signature changes — "
            "intra-module callers can be updated in the same edit. "
            "Call this before changing a function's parameters or return type."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "fqn": {"type": "string", "description": "Fully Qualified Name of the function to check."}
            },
            "required": ["fqn"],
        },
    },
    {
        "name": "get_file_interface",
        "description": (
            "Get the current public interface of all functions in a file — "
            "their names, parameter lists, and return types as stored in the graph. "
            "Call this before editing a file to know which function contracts exist "
            "and what would break downstream if they changed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Relative file path (e.g., fastapi/routing.py)."}
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "analyze_git_diff",
        "description": (
            "Given a git ref, find all external callers that break due to interface changes "
            "in that commit. Defaults to HEAD (most recent commit). "
            "Equivalent to running get_file_interface + analyze_edit_impact across all changed "
            "files at once, and also reports callers of deleted functions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ref": {
                    "type": "string",
                    "description": "Git ref to analyse (default: HEAD)."
                },
                "repo_root": {
                    "type": "string",
                    "description": "Repository root path (optional — auto-detected if omitted)."
                },
            },
            "required": [],
        },
    },
    {
        "name": "ingest_repository",
        "description": (
            "Parse a local repository and build its code knowledge graph in FalkorDB. "
            "Call this first when working with a new repo. Returns counts of functions, "
            "classes, and call edges discovered."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "repo_path": {
                    "type": "string",
                    "description": "Absolute or relative path to the repository root (default: current dir).",
                }
            },
            "required": [],
        },
    },
    {
        "name": "get_graph_summary",
        "description": (
            "Return a summary of what is currently in the code graph — total functions, "
            "classes, edges, and top modules by function count. Call this to understand "
            "what has been indexed before running other graph tools."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "run_failing_tests_and_localize",
        "description": (
            "Run specific failing tests, capture the stack trace, and map the failing "
            "functions back to their graph FQNs. This gives deterministic, ground-truth "
            "seed localization — far more precise than semantic search alone. "
            "Use this before get_blast_radius to find exact bug locations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "repo_path": {
                    "type": "string",
                    "description": "Path to the repository to run tests in.",
                },
                "test_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "pytest test IDs to run (e.g. ['tests/test_q.py::QTests::test_filter']).",
                },
            },
            "required": ["repo_path", "test_ids"],
        },
    },
    {
        "name": "get_issue_context",
        "description": (
            "Given a GitHub issue URL or plain issue text, find the most likely functions "
            "to fix using hybrid scoring (semantic similarity + name matching). "
            "For GitHub URLs, fetches the issue title and body automatically. "
            "Returns ranked candidate functions with graph paths showing why each is relevant."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "github_url": {
                    "type": "string",
                    "description": "GitHub issue URL (e.g. https://github.com/django/django/issues/1234).",
                },
                "issue_text": {
                    "type": "string",
                    "description": "Plain text issue description (used if github_url is not provided).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_coverage_guided_blast_radius",
        "description": (
            "Get a precision-filtered blast radius by intersecting the static call graph "
            "with dynamic test coverage. Only returns callers that were actually executed "
            "by the failing tests — eliminating ~60% of irrelevant nodes from the blast radius. "
            "Requires pytest-cov: pip install pytest-cov"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "repo_path": {
                    "type": "string",
                    "description": "Path to the repository.",
                },
                "test_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Failing test IDs to run for coverage.",
                },
                "fqn": {
                    "type": "string",
                    "description": "FQN of the function to compute blast radius for.",
                },
            },
            "required": ["fqn"],
        },
    },
    {
        "name": "get_architecture_diagram",
        "description": (
            "Return a Mermaid flowchart diagram of the repository's module-level architecture — "
            "which modules call, import, or inherit from which. "
            "Bob can embed this directly in responses or docs. "
            "Shows the top N strongest module relationships by edge weight."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "top_n": {
                    "type": "integer",
                    "description": "Max number of edges to show (default: 20). Use fewer for simpler diagrams.",
                }
            },
            "required": [],
        },
    },
    {
        "name": "get_function_fingerprints",
        "description": (
            "Return the compact structured fingerprint for each requested function FQN. "
            "Fingerprints contain: signature, behavior label (if generated), calls list, reads list, "
            "raises list, and caller count — all in ~30 tokens. Use this on blast-radius results to "
            "understand 20+ functions without reading source. For functions with no behavior label, "
            "call get_function_skeletons next, generate a label, then store it with store_behavior_labels."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "fqns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of FQNs to retrieve fingerprints for.",
                },
            },
            "required": ["fqns"],
        },
    },
    {
        "name": "get_function_skeletons",
        "description": (
            "Return a stripped AST skeleton (~38% of source tokens) for functions that lack a behavior label. "
            "The skeleton preserves control-flow conditions, return statements, raises, and variable names "
            "while stripping string literals, argument values, and import noise. "
            "Use this to generate a behavior label efficiently — pass the skeleton to the LLM, "
            "then cache the label with store_behavior_labels. Skeletons are cached in FalkorDB after first use."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "fqns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of FQNs to get skeletons for.",
                },
            },
            "required": ["fqns"],
        },
    },
    {
        "name": "store_behavior_labels",
        "description": (
            "Permanently cache behavior labels you generated for undocumented functions. "
            "Labels are injected into the fingerprint and stored in FalkorDB — future calls to "
            "get_function_fingerprints will return them for free. "
            "Format: one concise code-comment style line, e.g. 'Resolves aliased FQN via prefix matching'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "labels": {
                    "type": "object",
                    "description": "Mapping of FQN → behavior label string.",
                    "additionalProperties": {"type": "string"},
                },
            },
            "required": ["labels"],
        },
    },
    {
        "name": "analyze_edit_impact",
        "description": (
            "Given a list of functions whose signatures you changed, returns which external callers "
            "are now at risk of breaking. Only interface changes (parameter list or return type) are "
            "reported — body-only changes are filtered out since callers cannot observe them."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The file that was edited."
                },
                "changed_signatures": {
                    "type": "array",
                    "description": "List of functions with their old and new signatures.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "fqn":        {"type": "string"},
                            "old_params": {"type": "array", "items": {"type": "string"}},
                            "new_params": {"type": "array", "items": {"type": "string"}},
                            "old_return": {"type": ["string", "null"]},
                            "new_return": {"type": ["string", "null"]},
                        },
                        "required": ["fqn", "old_params", "new_params"],
                    },
                },
            },
            "required": ["file_path", "changed_signatures"],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool implementations (helper functions used by _TOOL_MAP below)
# ---------------------------------------------------------------------------

def _ingest_repository(args: dict, graph) -> dict:
    from ingest import run_ingestion
    repo_path = args.get("repo_path", ".")
    try:
        report = run_ingestion(str(Path(repo_path).resolve()))
        return {
            "status": "ingested",
            "repo_path": str(repo_path),
            "functions": report.get("functions", 0),
            "classes": report.get("classes", 0),
            "call_edges": report.get("call_edges", 0),
            "files_parsed": report.get("files_parsed", 0),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def _get_graph_summary(args: dict, graph) -> dict:
    try:
        from graph_health import get_graph_health
        return get_graph_health(graph)
    except Exception as e:
        return {"error": str(e)}


def _run_failing_tests_and_localize(args: dict, graph) -> dict:
    repo_path = args.get("repo_path", ".")
    test_ids = list(args.get("test_ids", []))
    if not test_ids:
        return {"error": "test_ids is required"}

    cmd = ["python", "-m", "pytest"] + test_ids + ["--tb=long", "-q", "--no-header"]
    try:
        result = subprocess.run(
            cmd, cwd=str(repo_path),
            capture_output=True, text=True, timeout=120,
        )
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return {"error": "Test execution timed out after 120s"}
    except Exception as e:
        return {"error": f"Test execution failed: {e}"}

    # Parse traceback frames: File "path", line N, in func_name
    trace_pattern = re.compile(r'File "([^"]+)", line (\d+), in (\w+)')
    matches = trace_pattern.findall(output)

    seeds = []
    seen: set[tuple] = set()
    repo_resolved = str(Path(repo_path).resolve())

    for file_path, line_no, func_name in reversed(matches):
        # Normalize to relative path
        try:
            rel_path = str(Path(file_path).relative_to(repo_resolved))
        except ValueError:
            rel_path = file_path

        # Skip test files — the bug is in the implementation, not the test
        rel_parts = Path(rel_path).parts
        if any(p in ("tests", "test", "testing") or p.startswith("test_") or p.endswith("_test.py")
               for p in rel_parts):
            continue

        key = (rel_path, func_name)
        if key in seen:
            continue
        seen.add(key)

        # Attempt to map to a graph FQN
        fqn_found = None
        file_path_found = rel_path
        try:
            # Exact match first
            rows = graph.query(
                "MATCH (f:Function {name: $name, file_path: $fp}) "
                "RETURN f.fqn, f.file_path LIMIT 1",
                {"name": func_name, "fp": rel_path},
            ).result_set
            if not rows:
                # Fallback: match by filename suffix (handles different base paths)
                filename = Path(rel_path).name
                rows = graph.query(
                    "MATCH (f:Function {name: $name}) "
                    "WHERE f.file_path ENDS WITH $fn "
                    "RETURN f.fqn, f.file_path LIMIT 1",
                    {"name": func_name, "fn": filename},
                ).result_set
            if rows:
                fqn_found = rows[0][0]
                file_path_found = rows[0][1]
        except Exception:
            pass

        seeds.append({
            "fqn": fqn_found,
            "file_path": file_path_found,
            "func_name": func_name,
            "line": int(line_no),
            "in_graph": fqn_found is not None,
        })

    return {
        "seeds": seeds[:10],
        "raw_trace": output[-3000:],
        "test_ids": test_ids,
        "return_code": result.returncode,
        "total_frames": len(matches),
    }


def _get_issue_context(args: dict, graph) -> dict:
    input_url = args.get("github_url", "")
    input_text = args.get("issue_text", "")

    issue_meta: dict = {}
    query_text = input_text

    # Fetch GitHub issue if URL provided
    if input_url:
        gh_match = re.search(
            r"github\.com/([^/]+)/([^/]+)/issues/(\d+)", input_url
        )
        if gh_match:
            owner, repo, issue_num = gh_match.groups()
            api_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_num}"
            headers = {"Accept": "application/vnd.github.v3+json"}
            token = os.getenv("GITHUB_TOKEN")
            if token:
                headers["Authorization"] = f"token {token}"
            try:
                import requests as _req
                r = _req.get(api_url, headers=headers, timeout=10)
                r.raise_for_status()
                data = r.json()
                issue_meta = {
                    "title": data.get("title", ""),
                    "number": issue_num,
                    "labels": [lb["name"] for lb in data.get("labels", [])],
                    "url": input_url,
                }
                query_text = f"{data.get('title', '')} {data.get('body', '')}"
            except Exception as e:
                issue_meta = {"fetch_error": str(e)}
                query_text = input_url  # fall back to raw URL text
        else:
            query_text = input_url

    if not query_text.strip():
        return {"error": "Provide github_url or issue_text"}

    # Semantic search
    from tools import semantic_search as _sem_search
    search_result = _sem_search(query=query_text, graph=graph, top_k=20)
    candidates = search_result.get("results", [])

    # KGCompass-inspired hybrid scoring: 75% semantic + 25% name-token overlap
    q_tokens = set(re.findall(r"\w+", query_text.lower()))

    for c in candidates:
        name = c.get("name") or c.get("fqn", "").split(".")[-1]
        # Split camelCase + snake_case into tokens
        name_tokens = set(re.findall(r"[a-z]+", re.sub(r"([A-Z])", r"_\1", name).lower()))
        lev = len(q_tokens & name_tokens) / max(len(name_tokens), 1)
        c["hybrid_score"] = round(0.75 * float(c.get("score", 0)) + 0.25 * lev, 4)

    candidates.sort(key=lambda x: x.get("hybrid_score", 0), reverse=True)

    return {
        "issue": issue_meta,
        "query_preview": query_text[:300],
        "candidates": candidates[:10],
    }


def _get_coverage_guided_blast_radius(args: dict, graph) -> dict:
    from tools import get_blast_radius as _blast

    fqn = args.get("fqn", "")
    repo_path = args.get("repo_path", ".")
    test_ids = list(args.get("test_ids", []))

    if not fqn:
        return {"error": "fqn is required"}

    # Get full static blast radius first
    blast = _blast(fqn=fqn, graph=graph)
    blast_nodes = blast.get("affected", [])

    if not blast_nodes or not test_ids:
        return {**blast, "coverage_filtered": False,
                "note": "Returning full blast radius (no test_ids provided)"}

    # Run tests with coverage
    cov_path = Path(tempfile.mkdtemp()) / "coverage.json"
    cmd = (
        ["python", "-m", "pytest"] + test_ids
        + [f"--cov={repo_path}", f"--cov-report=json:{cov_path}",
           "-q", "--no-header", "--tb=no"]
    )
    try:
        subprocess.run(cmd, cwd=str(repo_path), capture_output=True,
                       timeout=120, check=False)
    except Exception as e:
        return {**blast, "coverage_filtered": False,
                "coverage_note": f"Coverage run failed: {e}"}

    # Parse coverage.json → file → executed line numbers
    covered_lines: dict[str, set[int]] = {}
    try:
        with open(cov_path) as f:
            cov_data = json.load(f)
        repo_resolved = str(Path(repo_path).resolve())
        for file_path, file_data in cov_data.get("files", {}).items():
            try:
                rel = str(Path(file_path).relative_to(repo_resolved))
            except ValueError:
                rel = file_path
            executed = set(file_data.get("executed_lines", []))
            if executed:
                covered_lines[rel] = executed
    except Exception as e:
        return {**blast, "coverage_filtered": False,
                "coverage_note": f"Coverage parse failed: {e}"}

    # Filter: keep blast nodes whose start_line was executed
    filtered = []
    for node in blast_nodes:
        node_file = node.get("file_path", "")
        if node_file not in covered_lines:
            continue  # file not touched by failing tests — irrelevant
        try:
            rows = graph.query(
                "MATCH (f:Function {fqn: $fqn}) RETURN f.start_line",
                {"fqn": node.get("fqn", "")},
            ).result_set
            start_line = int(rows[0][0]) if rows and rows[0][0] is not None else None
        except Exception:
            start_line = None

        if start_line and start_line in covered_lines[node_file]:
            filtered.append({**node, "covered": True})

    return {
        "affected": filtered,
        "total_blast_radius": len(blast_nodes),
        "after_coverage_filter": len(filtered),
        "nodes_eliminated": len(blast_nodes) - len(filtered),
        "coverage_filtered": True,
        "seed_fqn": fqn,
    }


def _get_architecture_diagram(args: dict, graph) -> dict:
    from tools import get_macro_architecture
    top_n = int(args.get("top_n", 20))
    try:
        arch = get_macro_architecture(graph)
        edges = arch.get("modules", [])
        # Sort by weight, take top N
        edges = sorted(edges, key=lambda e: e["weight"], reverse=True)[:top_n]
        if not edges:
            return {"mermaid": "graph TD\n  %% No module edges found — run ingest_repository first", "edges": 0}

        # Build Mermaid flowchart
        lines = ["graph TD"]
        seen_nodes: set[str] = set()

        def _node_id(name: str) -> str:
            return re.sub(r"[^a-zA-Z0-9]", "_", name)

        for e in edges:
            src, tgt = e["source"], e["target"]
            sid, tid = _node_id(src), _node_id(tgt)
            types = e.get("types", [])
            weight = e["weight"]

            # Add node labels on first appearance
            if sid not in seen_nodes:
                lines.append(f'    {sid}["{src}"]')
                seen_nodes.add(sid)
            if tid not in seen_nodes:
                lines.append(f'    {tid}["{tgt}"]')
                seen_nodes.add(tid)

            # Edge label: type + weight
            label = "/".join(sorted(types)) + f" ×{weight}"
            arrow = "-->" if "CALLS" in types else "-.->"
            lines.append(f"    {sid} {arrow}|{label}| {tid}")

        mermaid = "\n".join(lines)
        return {
            "mermaid": mermaid,
            "edges": len(edges),
            "note": "Paste into any Mermaid renderer or GitHub markdown to visualize.",
        }
    except Exception as e:
        return {"error": str(e)}


def _get_function_fingerprints(args: dict, graph) -> dict:
    fqns = list(args.get("fqns", []))
    if not fqns:
        return {"error": "fqns list is required"}

    # Single batch query instead of N round-trips
    result: dict[str, str | None] = {fqn: None for fqn in fqns}
    try:
        rows = graph.query(
            "UNWIND $fqns AS fqn "
            "MATCH (f:Function {fqn: fqn}) "
            "RETURN f.fqn, f.fingerprint",
            {"fqns": fqns},
        ).result_set
        for row in rows:
            if row[0] and row[1]:
                result[row[0]] = row[1]
    except Exception as e:
        logger.warning("fingerprint batch query failed: %s", e)

    found = sum(1 for v in result.values() if v is not None)
    return {"fingerprints": result, "found": found, "missing": len(fqns) - found}


def _get_function_skeletons(args: dict, graph) -> dict:
    from fingerprinting import build_code_skeleton
    fqns = list(args.get("fqns", []))
    if not fqns:
        return {"error": "fqns list is required"}

    # Batch-fetch all metadata in one query
    meta_map: dict[str, tuple] = {}
    try:
        rows = graph.query(
            "UNWIND $fqns AS fqn "
            "MATCH (f:Function {fqn: fqn}) "
            "RETURN f.fqn, f.skeleton, f.file_path, f.start_line, f.end_line",
            {"fqns": fqns},
        ).result_set
        for row in rows:
            meta_map[row[0]] = (row[1], row[2], row[3], row[4])  # skeleton, file_path, start, end
    except Exception as e:
        logger.warning("skeleton batch query failed: %s", e)

    result: dict[str, str | None] = {}
    for fqn in fqns:
        if fqn not in meta_map:
            result[fqn] = None
            continue
        try:
            skeleton, file_path, start_line, end_line = meta_map[fqn]

            if skeleton:
                result[fqn] = skeleton
                continue

            if not file_path or start_line is None or end_line is None or int(start_line) < 1:
                result[fqn] = None
                continue

            try:
                src_lines = Path(file_path).read_text(errors="replace").splitlines()
                src = "\n".join(src_lines[int(start_line) - 1 : int(end_line)])
                skel = build_code_skeleton(src)
                result[fqn] = skel
                try:
                    graph.query(
                        "MATCH (f:Function {fqn: $fqn}) SET f.skeleton = $sk",
                        {"fqn": fqn, "sk": skel},
                    )
                except Exception:
                    pass
            except Exception as e:
                logger.warning("skeleton build failed for %s: %s", fqn, e)
                result[fqn] = None
        except Exception as e:
            logger.warning("skeleton processing failed for %s: %s", fqn, e)
            result[fqn] = None

    found = sum(1 for v in result.values() if v is not None)
    return {"skeletons": result, "found": found, "missing": len(fqns) - found}


def _store_behavior_labels(args: dict, graph) -> dict:
    from fingerprinting import inject_behavior_label
    labels: dict[str, str] = dict(args.get("labels", {}))
    if not labels:
        return {"error": "labels dict is required"}

    stored = 0
    errors: list[str] = []
    for fqn, label in labels.items():
        try:
            rows = graph.query(
                "MATCH (f:Function {fqn: $fqn}) RETURN f.fingerprint",
                {"fqn": fqn},
            ).result_set
            if not rows:
                errors.append(f"{fqn}: not found in graph")
                continue

            current_fp = rows[0][0] or ""
            updated_fp = inject_behavior_label(current_fp, label)
            graph.query(
                "MATCH (f:Function {fqn: $fqn}) SET f.fingerprint = $fp, f.fingerprint_label = $label",
                {"fqn": fqn, "fp": updated_fp, "label": label},
            )
            stored += 1
        except Exception as e:
            errors.append(f"{fqn}: {e}")
            logger.warning("store_behavior_label failed for %s: %s", fqn, e)

    return {"stored": stored, "total": len(labels), "errors": errors}


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_TOOL_MAP = {
    "get_function_context": lambda args, g: get_function_context(fqn=args["fqn"], graph=g),
    "get_callers": lambda args, g: get_callers(fqn=args["fqn"], graph=g),
    "get_callees": lambda args, g: get_callees(fqn=args["fqn"], graph=g),
    "get_impact_radius": lambda args, g: get_impact_radius(fqn=args["fqn"], graph=g),
    "get_blast_radius": lambda args, g: get_blast_radius(fqn=args["fqn"], graph=g),
    "get_source_code": lambda args, g: get_source_code(fqn=args["fqn"], graph=g),
    "semantic_search": lambda args, g: semantic_search(query=args["query"], graph=g),
    "get_macro_architecture":    lambda args, g: get_macro_architecture(graph=g),
    "get_class_architecture":    lambda args, g: get_class_architecture(module_name=args["module_name"], graph=g),
    "get_cross_module_callers":  lambda args, g: get_cross_module_callers(fqn=args["fqn"], graph=g),
    "get_file_interface":        lambda args, g: get_file_interface(file_path=args["file_path"], graph=g),
    "get_module_readers":        lambda args, g: get_module_readers(module_name=args["module_name"], graph=g),
    "analyze_git_diff":          lambda args, g: git_diff_impact(
                                     ref=args.get("ref", "HEAD"),
                                     repo_root=args.get("repo_root"),
                                     graph=g,
                                 ),
    "analyze_edit_impact":       lambda args, g: analyze_edit_impact(
                                     file_path=args["file_path"],
                                     changed_signatures=args["changed_signatures"],
                                     graph=g,
                                 ),
    "get_architecture_diagram":         _get_architecture_diagram,
    "get_function_fingerprints":        _get_function_fingerprints,
    "get_function_skeletons":           _get_function_skeletons,
    "store_behavior_labels":            _store_behavior_labels,
    "ingest_repository":                _ingest_repository,
    "get_graph_summary":                _get_graph_summary,
    "run_failing_tests_and_localize":   _run_failing_tests_and_localize,
    "get_issue_context":                _get_issue_context,
    "get_coverage_guided_blast_radius": _get_coverage_guided_blast_radius,
}


def _dispatch(tool_name: str, arguments: dict[str, Any], graph) -> str:
    """Route an MCP tool call to the corresponding tools.py function."""
    handler = _TOOL_MAP.get(tool_name)
    if handler is None:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    try:
        result = handler(arguments, graph)
        return json.dumps(result)
    except Exception as e:
        logger.error("MCP tool '%s' failed: %s", tool_name, e)
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# MCP server setup
# ---------------------------------------------------------------------------

def create_mcp_server() -> "Server":
    """Build and configure the MCP server with all Repo-Insight tools."""
    if not MCP_AVAILABLE:
        raise ImportError(
            "The 'mcp' package is required. Install it with: pip install mcp"
        )

    app = Server("repo-insight")
    graph = get_connection()

    @app.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name=td["name"],
                description=td["description"],
                inputSchema=td["input_schema"],
            )
            for td in TOOL_DEFINITIONS
        ]

    @app.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        result_str = _dispatch(name, arguments, graph)
        return [TextContent(type="text", text=result_str)]

    return app


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

async def _run_stdio(app: "Server") -> None:
    """Run the MCP server over stdio."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repo-Insight MCP Server — expose graph tools to AI coding assistants",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio"],
        default="stdio",
        help="Transport to use (default: stdio)",
    )
    args = parser.parse_args()

    if not MCP_AVAILABLE:
        print("ERROR: MCP SDK not installed. Run: pip install mcp")
        return

    import asyncio
    app = create_mcp_server()
    print("Repo-Insight MCP server starting (stdio transport)...")
    asyncio.run(_run_stdio(app))


if __name__ == "__main__":
    main()
