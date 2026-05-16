#!/usr/bin/env python3
"""
SWE-bench Verified Mini harness — agentic LLM loop with Repo-Insight MCP tools.

Dataset: MariusHobbhahn/swe-bench-verified-mini (50 instances, Princeton HAL leaderboard)
Uses the same Django/Sphinx repos as SWE-bench Verified — ideal for graph-based tools.

Two modes (A/B ablation built-in):
  --no-graph   Baseline: agent uses only write_file + run_tests (no Repo-Insight tools)
  (default)    Graph-enhanced: agent has all 22 Repo-Insight tools available

Headline metrics reported:
  - Resolution rate (% resolved, via official SWE-bench evaluator on predictions.jsonl)
  - Tokens per resolved instance (input+output, averaged over solved cases)
  - Tokens per attempted instance (all cases)

Setup:
  1. Start vLLM:
       python -m vllm.entrypoints.openai.api_server \\
         --model Qwen/Qwen3-32B --port 8000
  2. Start FalkorDB:
       docker compose -f docker-compose.local.yml up -d
  3. Run graph-enhanced condition:
       AGENT_BASE_URL=http://localhost:8000/v1 AGENT_MODEL=Qwen/Qwen3-32B \\
       SKIP_JEDI=true python run_swebench_claude.py --workers 4 --output-dir ./results/graph
  4. Run no-graph baseline:
       AGENT_BASE_URL=http://localhost:8000/v1 AGENT_MODEL=Qwen/Qwen3-32B \\
       python run_swebench_claude.py --no-graph --workers 4 --output-dir ./results/baseline

Environment variables:
  AGENT_BASE_URL   OpenAI-compatible endpoint (vLLM/Ollama/LiteLLM). Required.
  AGENT_MODEL      Model name as served by the endpoint. Required.
  AGENT_API_KEY    API key (default: EMPTY)
  AGENT_MAX_ITER   Max tool-call iterations per instance (default: 20)
  SKIP_JEDI        Set to 'true' to skip Jedi call resolution (faster ingest)
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from datasets import load_dataset
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration — all via env vars, no SGLang defaults
# ---------------------------------------------------------------------------

AGENT_BASE_URL  = os.environ.get("AGENT_BASE_URL", "")        # required
AGENT_MODEL     = os.environ.get("AGENT_MODEL", "")           # required
AGENT_API_KEY   = os.environ.get("AGENT_API_KEY", "EMPTY")
MAX_ITERATIONS  = int(os.environ.get("AGENT_MAX_ITER", "20"))

# SWE-bench Verified Mini — 50 instances, Princeton HAL leaderboard
# Drop-in replacement for SWE-bench Verified; django + sphinx repos only
SWEBENCH_DATASET = "MariusHobbhahn/swe-bench-verified-mini"

CLONE_TIMEOUT    = 120    # seconds for git clone
TOOL_TIMEOUT     = 120    # seconds per tool call (tests, ingest)
INSTANCE_TIMEOUT = 1800   # 30 min hard cap per instance

_REPO_URL_TEMPLATE = "https://github.com/{repo}.git"

# FalkorDB connection settings (read from env, same as config.py)
FALKORDB_HOST = os.environ.get("FALKORDB_HOST", "localhost")
FALKORDB_PORT = int(os.environ.get("FALKORDB_PORT", "6379"))
GRAPH_NAME    = os.environ.get("GRAPH_NAME", "repo_insight")

logger = logging.getLogger("swebench_agent")


# ---------------------------------------------------------------------------
# Git helpers (identical logic to run_swebench.py)
# ---------------------------------------------------------------------------

def _clone_repo(repo: str, commit: str, dest: Path) -> None:
    repo_url = _REPO_URL_TEMPLATE.format(repo=repo)
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", str(dest)],
                   check=True, capture_output=True, timeout=10)
    subprocess.run(["git", "remote", "add", "origin", repo_url],
                   cwd=str(dest), check=True, capture_output=True, timeout=10)
    subprocess.run(["git", "fetch", "--depth=1", "origin", commit],
                   cwd=str(dest), check=True, capture_output=True, timeout=CLONE_TIMEOUT)
    subprocess.run(["git", "checkout", "FETCH_HEAD"],
                   cwd=str(dest), check=True, capture_output=True, timeout=30)


def _capture_diff(repo_dir: Path) -> str:
    result = subprocess.run(
        ["git", "diff", "HEAD"], cwd=str(repo_dir),
        capture_output=True, text=True, timeout=30,
    )
    return result.stdout


def _validate_patch(repo_dir: Path) -> tuple[bool, str]:
    changed = subprocess.run(
        ["git", "diff", "--name-only", "HEAD"],
        cwd=str(repo_dir), capture_output=True, text=True, timeout=30,
    ).stdout.splitlines()
    for rel in changed:
        if not rel.endswith(".py"):
            continue
        try:
            import ast
            ast.parse((repo_dir / rel).read_text(errors="replace"))
        except SyntaxError as e:
            return False, f"{rel}: {e}"
    return True, ""


# ---------------------------------------------------------------------------
# OpenAI tool definitions for the 22 Repo-Insight tools
# ---------------------------------------------------------------------------

# Baseline tools — no graph intelligence, just file edit + test run
_BASELINE_TOOL_DEFS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write the full content of a file in the repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run pytest tests to verify your fix.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_path": {"type": "string"},
                    "test_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["repo_path", "test_ids"],
            },
        },
    },
]

_TOOL_DEFS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "ingest_repository",
            "description": (
                "Parse the cloned repository and build a code knowledge graph in FalkorDB. "
                "Call this FIRST — all other graph tools require the graph to be built."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_path": {
                        "type": "string",
                        "description": "Absolute path to the cloned repo directory.",
                    }
                },
                "required": ["repo_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_failing_tests_and_localize",
            "description": (
                "Run the failing tests and parse the stack trace to find the exact functions "
                "where the bug lives. Returns graph FQNs for the deepest failing frames. "
                "Use BEFORE get_blast_radius — this gives ground-truth seeds."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_path": {"type": "string"},
                    "test_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "pytest test IDs from FAIL_TO_PASS list.",
                    },
                },
                "required": ["repo_path", "test_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_blast_radius",
            "description": (
                "Return all functions that transitively CALL the given function (upstream). "
                "Results include call chain paths. Use to understand the impact of changing fqn."
            ),
            "parameters": {
                "type": "object",
                "properties": {"fqn": {"type": "string"}},
                "required": ["fqn"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_function_fingerprints",
            "description": (
                "Return compact structured fingerprints for a batch of FQNs. "
                "Each fingerprint is ~30 tokens: signature + behavior label + calls + raises + caller count. "
                "Call this on blast-radius FQNs before deciding which to read fully — saves 15× tokens."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fqns": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["fqns"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_function_skeletons",
            "description": (
                "Return AST skeletons (~38% of source tokens) for undocumented functions. "
                "Preserves conditions, returns, raises — strips noise. "
                "Use when a fingerprint lacks a behavior label and you need structural context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fqns": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["fqns"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "store_behavior_labels",
            "description": (
                "Cache behavior labels you generated for undocumented functions. "
                "Labels are stored in FalkorDB and returned by get_function_fingerprints forever after."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "labels": {
                        "type": "object",
                        "description": "Mapping of FQN → one-line behavior description.",
                        "additionalProperties": {"type": "string"},
                    }
                },
                "required": ["labels"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_source_code",
            "description": (
                "Read the full source of a function or class by FQN. "
                "Use sparingly — only for the 1-3 functions you actually need to modify."
            ),
            "parameters": {
                "type": "object",
                "properties": {"fqn": {"type": "string"}},
                "required": ["fqn"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_callers",
            "description": "Direct callers of a function.",
            "parameters": {
                "type": "object",
                "properties": {"fqn": {"type": "string"}},
                "required": ["fqn"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_callees",
            "description": "Direct callees of a function.",
            "parameters": {
                "type": "object",
                "properties": {"fqn": {"type": "string"}},
                "required": ["fqn"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "semantic_search",
            "description": (
                "Find functions by natural language description. "
                "Use when you know what the buggy behavior is but not which function causes it."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_function_context",
            "description": "File path, module, and summary for a function by FQN.",
            "parameters": {
                "type": "object",
                "properties": {"fqn": {"type": "string"}},
                "required": ["fqn"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write the full content of a file in the repository. "
                "Use to apply your bug fix. Always write the complete file — not just the diff."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file to write.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Complete file content.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": (
                "Run pytest tests in the repository to verify your fix. "
                "Always run the FAIL_TO_PASS tests after writing your fix."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_path": {"type": "string"},
                    "test_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Test IDs to run. Use the FAIL_TO_PASS list.",
                    },
                },
                "required": ["repo_path", "test_ids"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool executor — routes agent tool calls to Repo-Insight implementations
# ---------------------------------------------------------------------------

def _execute_tool(
    name: str,
    args: dict,
    graph,
    repo_path: Path,
    graph_name: str,
) -> Any:
    from mcp_server import _TOOL_MAP
    from ingest import run_ingestion

    # ---- file editing ----
    if name == "write_file":
        path = Path(args["path"])
        if not path.is_absolute():
            path = repo_path / path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(args["content"])
            return {"status": "written", "path": str(path), "bytes": len(args["content"])}
        except Exception as e:
            return {"error": str(e)}

    # ---- test execution ----
    if name == "run_tests":
        rp = args.get("repo_path") or str(repo_path)
        test_ids = list(args.get("test_ids", []))
        cmd = ["python", "-m", "pytest"] + test_ids + ["--tb=short", "-q", "--no-header"]
        try:
            proc = subprocess.run(
                cmd, cwd=rp, capture_output=True, text=True, timeout=TOOL_TIMEOUT,
            )
            output = (proc.stdout + proc.stderr)[-4000:]
            return {"returncode": proc.returncode, "output": output, "passed": proc.returncode == 0}
        except subprocess.TimeoutExpired:
            return {"error": f"Tests timed out after {TOOL_TIMEOUT}s", "passed": False}
        except Exception as e:
            return {"error": str(e), "passed": False}

    # ---- ingest (needs graph_name injected) ----
    if name == "ingest_repository":
        rp = args.get("repo_path") or str(repo_path)
        try:
            report = run_ingestion(str(Path(rp).resolve()), graph_name=graph_name)
            return {"status": "ingested", **report}
        except Exception as e:
            return {"error": str(e)}

    # ---- run_failing_tests_and_localize (inject repo_path default) ----
    if name == "run_failing_tests_and_localize":
        merged = dict(args)
        if "repo_path" not in merged:
            merged["repo_path"] = str(repo_path)
        try:
            return _TOOL_MAP[name](merged, graph)
        except Exception as e:
            return {"error": str(e)}

    # ---- all other Repo-Insight tools ----
    handler = _TOOL_MAP.get(name)
    if handler is None:
        return {"error": f"Unknown tool: {name}"}
    try:
        return handler(args, graph)
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert Python software engineer fixing bugs in open-source repositories.
You have access to Repo-Insight — a code knowledge graph with 22 analysis tools.

Recommended workflow:
1. ingest_repository(repo_path)                    — build the code graph (do this first)
2. run_failing_tests_and_localize(repo_path, tests) — find the exact failing function
3. get_function_fingerprints([fqns])               — understand blast radius cheaply (~30 tokens each)
4. get_source_code(fqn)                            — read full source of 1-3 key functions only
5. write_file(path, full_content)                  — apply your fix
6. run_tests(repo_path, tests)                     — verify fix passes

Rules:
- Make the minimum change to fix the bug. Do not refactor or rename.
- Write the COMPLETE file content when using write_file — not a diff.
- Do not edit test files.
- After writing a fix, always run the FAIL_TO_PASS tests to verify.
- If tests fail, read the output carefully, revise, and try again.
"""


def _run_agent_loop(
    instance: dict,
    graph,
    repo_path: Path,
    graph_name: str,
    client: OpenAI,
    instance_id: str,
    no_graph: bool = False,
) -> tuple[str, dict]:
    problem = instance.get("problem_statement", "")
    fail_to_pass = json.loads(instance.get("FAIL_TO_PASS", "[]"))
    ftp_str = "\n".join(f"  - {t}" for t in fail_to_pass[:10])

    # Baseline: only file-edit + test-run tools (no graph intelligence)
    active_tools = _BASELINE_TOOL_DEFS if no_graph else _TOOL_DEFS

    system = _SYSTEM_PROMPT if not no_graph else (
        "You are an expert Python software engineer fixing bugs in open-source repositories.\n"
        "You can write files and run tests. Fix the bug so the FAIL_TO_PASS tests pass.\n"
        "Rules: minimal change only; write complete file content; do not edit test files."
    )

    messages: list[dict] = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": (
                f"Fix the following bug.\n\n"
                f"Repository path: {repo_path}\n"
                + ("" if no_graph else f"FalkorDB graph name: {graph_name}\n")
                + f"\nProblem statement:\n{problem[:3000]}\n\n"
                f"Failing tests (FAIL_TO_PASS — these must pass after your fix):\n{ftp_str}"
            ),
        },
    ]

    token_stats = {"input_tokens": 0, "output_tokens": 0, "iterations": 0}

    for iteration in range(MAX_ITERATIONS):
        try:
            response = client.chat.completions.create(
                model=AGENT_MODEL,
                messages=messages,
                tools=active_tools,
                tool_choice="auto",
                max_tokens=4096,
                temperature=0,
            )
        except Exception as e:
            logger.warning("[%s] LLM call failed iter=%d: %s", instance_id, iteration, e)
            break

        # Accumulate token counts
        if response.usage:
            token_stats["input_tokens"]  += response.usage.prompt_tokens
            token_stats["output_tokens"] += response.usage.completion_tokens
        token_stats["iterations"] = iteration + 1

        choice = response.choices[0]
        msg = choice.message

        # Serialise assistant turn
        assistant_turn: dict = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_turn["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_turn)

        # Stop if no tool calls (agent decided it's done)
        if not msg.tool_calls:
            logger.info("[%s] Agent finished after %d iterations", instance_id, iteration + 1)
            break

        # Execute each tool call and feed result back
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                fn_args = {}

            logger.info("[%s] iter=%d  %s(%s)", instance_id, iteration, fn_name,
                        str(list(fn_args.keys()))[:60])

            tool_result = _execute_tool(fn_name, fn_args, graph, repo_path, graph_name)
            # Truncate large results so we don't overflow context
            result_str = json.dumps(tool_result)
            if len(result_str) > 8000:
                result_str = result_str[:8000] + "... [truncated]"

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_str,
            })

    return _capture_diff(repo_path), token_stats


# ---------------------------------------------------------------------------
# Per-instance runner
# ---------------------------------------------------------------------------

def _run_instance(
    instance: dict,
    output_dir: Path,
    skip_existing: bool = False,
    graph_name: Optional[str] = None,
    no_graph: bool = False,
) -> dict:
    from ingest import get_connection

    instance_id = instance["instance_id"]
    repo        = instance["repo"]
    base_commit = instance["base_commit"]

    if graph_name is None:
        graph_name = GRAPH_NAME

    result: dict = {
        "instance_id": instance_id,
        "repo": repo,
        "status": "pending",
        "patch": "",
        "patch_file": None,
        "error": None,
        "duration_s": 0.0,
        "syntax_valid": None,
        "model": AGENT_MODEL,
        "no_graph": no_graph,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "iterations": 0,
    }

    patch_path = output_dir / f"{instance_id}.patch"
    if skip_existing and patch_path.exists():
        result.update(status="skipped", patch_file=str(patch_path))
        logger.info("[%s] Skipping — patch exists", instance_id)
        return result

    t0 = time.monotonic()
    tmp_root: Optional[Path] = None
    try:
        tmp_root = Path(tempfile.mkdtemp(prefix=f"swe_{instance_id}_"))
        repo_dir = tmp_root / "repo"

        logger.info("[%s] Cloning %s @ %s", instance_id, repo, base_commit[:10])
        _clone_repo(repo, base_commit, repo_dir)

        graph = get_connection(graph_name)
        client = OpenAI(base_url=AGENT_BASE_URL, api_key=AGENT_API_KEY)

        logger.info("[%s] Starting agent loop (max_iter=%d, no_graph=%s)",
                    instance_id, MAX_ITERATIONS, no_graph)
        patch, tok = _run_agent_loop(
            instance, graph, repo_dir, graph_name, client, instance_id, no_graph=no_graph
        )

        result.update(
            input_tokens=tok["input_tokens"],
            output_tokens=tok["output_tokens"],
            total_tokens=tok["input_tokens"] + tok["output_tokens"],
            iterations=tok["iterations"],
        )
        result["patch"] = patch
        result["duration_s"] = round(time.monotonic() - t0, 1)

        if patch.strip():
            patch_path.write_text(patch)
            result["patch_file"] = str(patch_path)
            valid, err = _validate_patch(repo_dir)
            result["syntax_valid"] = valid
            result["status"] = "patched" if valid else "syntax_error"
            if not valid:
                result["error"] = err
        else:
            result["status"] = "empty_diff"

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        result["duration_s"] = round(time.monotonic() - t0, 1)
        logger.error("[%s] Failed: %s", instance_id, e, exc_info=True)
    finally:
        if tmp_root and tmp_root.exists():
            shutil.rmtree(tmp_root, ignore_errors=True)

    logger.info("[%s] %s  %.1fs  patch=%d chars",
                instance_id, result["status"], result["duration_s"], len(result["patch"]))
    return result


# ---------------------------------------------------------------------------
# Sequential / parallel runners
# ---------------------------------------------------------------------------

def _run_sequential(
    dataset: list[dict],
    output_dir: Path,
    skip_existing: bool,
    graph_name: Optional[str] = None,
    no_graph: bool = False,
) -> list[dict]:
    results = []
    for i, instance in enumerate(dataset):
        logger.info("--- [%d/%d] %s ---", i + 1, len(dataset), instance["instance_id"])
        results.append(_run_instance(instance, output_dir, skip_existing, graph_name, no_graph))
    return results


def _worker_chunk(args: tuple) -> list[dict]:
    chunk, output_dir_str, skip_existing, worker_id, no_graph = args
    output_dir = Path(output_dir_str)
    gname = f"{GRAPH_NAME}_agent_w{worker_id}"
    logging.basicConfig(level=logging.INFO,
                        format=f"%(asctime)s [W{worker_id}] %(message)s")
    return _run_sequential(chunk, output_dir, skip_existing, gname, no_graph)


def _run_parallel(
    dataset: list[dict],
    output_dir: Path,
    skip_existing: bool,
    n_workers: int,
    no_graph: bool = False,
) -> list[dict]:
    chunk_size = (len(dataset) + n_workers - 1) // n_workers
    chunks = [dataset[i : i + chunk_size] for i in range(0, len(dataset), chunk_size)]
    args = [(c, str(output_dir), skip_existing, i, no_graph) for i, c in enumerate(chunks)]

    all_results: list[dict] = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_worker_chunk, a): a[3] for a in args}
        for fut in as_completed(futures):
            try:
                all_results.extend(fut.result())
            except Exception as e:
                logger.error("Worker %d failed: %s", futures[fut], e)
    return all_results


# ---------------------------------------------------------------------------
# Summary + predictions.jsonl
# ---------------------------------------------------------------------------

def _write_predictions(results: list[dict], output_dir: Path) -> Path:
    pred_path = output_dir / "predictions.jsonl"
    model_tag = f"agent-{AGENT_MODEL.split('/')[-1]}"
    with open(pred_path, "w") as f:
        for r in results:
            patch = r.get("patch", "")
            if not patch and r.get("patch_file"):
                try:
                    patch = Path(r["patch_file"]).read_text()
                except Exception:
                    pass
            f.write(json.dumps({
                "instance_id": r["instance_id"],
                "model_patch": patch,
                "model_name_or_path": model_tag,
            }) + "\n")
    return pred_path


def _print_summary(results: list[dict], output_dir: Path) -> None:
    counts: dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    total    = len(results)
    patched  = counts.get("patched", 0)
    skipped  = counts.get("skipped", 0)
    empty    = counts.get("empty_diff", 0)
    errors   = counts.get("error", 0) + counts.get("syntax_error", 0)
    non_skip = [r for r in results if r["status"] != "skipped"]
    avg_dur  = sum(r["duration_s"] for r in non_skip) / max(len(non_skip), 1)
    no_graph = results[0].get("no_graph", False) if results else False

    # Token efficiency metrics — the core claim
    resolved = [r for r in results if r["status"] == "patched"]
    attempted = [r for r in results if r["status"] != "skipped"]
    avg_tok_resolved  = (sum(r["total_tokens"] for r in resolved)  / max(len(resolved), 1))
    avg_tok_attempted = (sum(r["total_tokens"] for r in attempted) / max(len(attempted), 1))

    condition = "BASELINE (no graph)" if no_graph else "GRAPH-ENHANCED (Repo-Insight)"
    print(f"\n{'='*55}")
    print(f"  SWE-bench Verified Mini — {condition}")
    print(f"{'='*55}")
    print(f"Dataset: {SWEBENCH_DATASET}")
    print(f"Model:   {AGENT_MODEL}")
    print(f"")
    print(f"Resolution rate:   {patched}/{total}  ({100*patched/max(total,1):.1f}%)")
    print(f"Patched:           {patched}")
    print(f"Empty diff:        {empty}")
    print(f"Errors:            {errors}")
    print(f"Skipped:           {skipped}")
    print(f"")
    print(f"Avg tokens/resolved:  {avg_tok_resolved:,.0f}")
    print(f"Avg tokens/attempted: {avg_tok_attempted:,.0f}")
    print(f"Avg time/instance:    {avg_dur:.0f}s")

    pred_path = _write_predictions(results, output_dir)
    (output_dir / "results.json").write_text(json.dumps(results, indent=2))

    print(f"\nPredictions: {pred_path}")
    print(f"Results dir: {output_dir}")
    print("\nNext: run the official SWE-bench evaluator on predictions.jsonl")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / f"agent_{ts}.log"),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SWE-bench Lite — agentic loop with Repo-Insight MCP tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Max instances to run (default: all 300)")
    parser.add_argument("--instances", type=str, default=None,
                        help="Comma-separated instance IDs to run")
    parser.add_argument("--output-dir", default="./results/agent_v1",
                        help="Directory for patches, logs, predictions.jsonl")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip instances that already have a .patch file")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel worker processes (default: 1)")
    parser.add_argument("--no-graph", action="store_true",
                        help="Baseline condition: disable all Repo-Insight tools (write_file + run_tests only)")
    args = parser.parse_args()

    # Validate required env vars
    if not AGENT_BASE_URL:
        print("ERROR: AGENT_BASE_URL is required. Example:")
        print("  export AGENT_BASE_URL=http://localhost:8000/v1")
        sys.exit(1)
    if not AGENT_MODEL:
        print("ERROR: AGENT_MODEL is required. Example:")
        print("  export AGENT_MODEL=Qwen/Qwen3-32B")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(output_dir / "logs")

    logger.info("Model:      %s", AGENT_MODEL)
    logger.info("Endpoint:   %s", AGENT_BASE_URL)
    logger.info("Max iter:   %d", MAX_ITERATIONS)
    logger.info("Workers:    %d", args.workers)

    logger.info("Loading %s …", SWEBENCH_DATASET)
    ds = load_dataset(SWEBENCH_DATASET, split="test")
    dataset = list(ds)

    if args.instances:
        ids = {i.strip() for i in args.instances.split(",")}
        dataset = [d for d in dataset if d["instance_id"] in ids]
        logger.info("Filtered to %d instances: %s", len(dataset), args.instances[:80])
    if args.limit:
        dataset = dataset[: args.limit]

    logger.info("Running %d instances", len(dataset))

    no_graph = args.no_graph
    logger.info("Condition: %s", "BASELINE (no graph)" if no_graph else "GRAPH-ENHANCED")

    n = max(1, args.workers)
    if n == 1:
        results = _run_sequential(dataset, output_dir, args.skip_existing, no_graph=no_graph)
    else:
        results = _run_parallel(dataset, output_dir, args.skip_existing, n, no_graph=no_graph)

    _print_summary(results, output_dir)


if __name__ == "__main__":
    main()
