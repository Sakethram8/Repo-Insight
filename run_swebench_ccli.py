#!/usr/bin/env python3
"""
SWE-bench Verified Mini harness using Claude Code CLI as the agent.

Claude Code (with vLLM/Qwen via LiteLLM proxy) is the agent.
Repo-Insight MCP tools are injected via .claude/mcp.json in each cloned repo.
The A/B ablation is controlled by --no-graph: same agent, same model, only the tools differ.

Setup:
  1. Install Claude Code CLI:
       npm install -g @anthropic-ai/claude-code

  2. Start vLLM:
       python -m vllm.entrypoints.openai.api_server \\
         --model Qwen/Qwen3-32B --port 8000 --tensor-parallel-size 2

  3. Start LiteLLM proxy (bridges Claude Code → vLLM):
       litellm --config litellm_config.yaml --port 4000

  4. Start FalkorDB:
       docker compose -f docker-compose.local.yml up -d

  5. Run graph-enhanced condition:
       ANTHROPIC_BASE_URL=http://localhost:4000 ANTHROPIC_API_KEY=fake \\
       SKIP_JEDI=true python run_swebench_ccli.py --workers 4 --output-dir ./results/ccli_graph

  6. Run no-graph baseline:
       ANTHROPIC_BASE_URL=http://localhost:4000 ANTHROPIC_API_KEY=fake \\
       python run_swebench_ccli.py --no-graph --workers 4 --output-dir ./results/ccli_baseline

Headline metrics:
  - Resolution rate via official SWE-bench evaluator on predictions.jsonl
  - Tokens per resolved instance (parsed from Claude Code output)
"""

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

from datasets import load_dataset

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SWEBENCH_DATASET = "MariusHobbhahn/swe-bench-verified-mini"

REPO_INSIGHT_DIR = Path(__file__).parent.resolve()
PYTHON_BIN       = sys.executable           # absolute path for MCP config
MCP_SERVER_PATH  = REPO_INSIGHT_DIR / "mcp_server.py"

FALKORDB_HOST = os.environ.get("FALKORDB_HOST", "localhost")
FALKORDB_PORT = os.environ.get("FALKORDB_PORT", "6379")
GRAPH_NAME    = os.environ.get("GRAPH_NAME", "repo_insight")

# Model name must match --served-model-name in vLLM exactly.
# Claude Code defaults to claude-opus-4-7; override with this env var.
CLAUDE_MODEL  = os.environ.get("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")

CLONE_TIMEOUT   = 120    # seconds for git clone
AGENT_TIMEOUT   = 1200   # 20 min cap per instance for Claude Code
INSTANCE_TIMEOUT = 1500  # 25 min total per instance

_REPO_URL_TEMPLATE = "https://github.com/{repo}.git"

logger = logging.getLogger("swebench_ccli")


# ---------------------------------------------------------------------------
# Git helpers (shared with run_swebench_claude.py)
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
# MCP config generation
# ---------------------------------------------------------------------------

def _write_mcp_config(repo_dir: Path, graph_name: str, no_graph: bool) -> None:
    """Write .mcp.json into the cloned repo root (Claude Code project-scope format)."""
    if no_graph:
        config = {"mcpServers": {}}
    else:
        config = {
            "mcpServers": {
                "repo-insight": {
                    "command": str(PYTHON_BIN),
                    "args": [str(MCP_SERVER_PATH)],
                    "env": {
                        "FALKORDB_HOST": FALKORDB_HOST,
                        "FALKORDB_PORT": FALKORDB_PORT,
                        "GRAPH_NAME": graph_name,
                        "SKIP_JEDI": os.environ.get("SKIP_JEDI", "false"),
                    },
                }
            }
        }

    (repo_dir / ".mcp.json").write_text(json.dumps(config, indent=2))


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(instance: dict, repo_path: Path, graph_name: str, no_graph: bool) -> str:
    problem   = instance.get("problem_statement", "")[:3000]
    fail_list = json.loads(instance.get("FAIL_TO_PASS", "[]"))
    ftp_str   = "\n".join(f"  - {t}" for t in fail_list[:10])

    base = (
        f"Fix the bug described below in the repository at {repo_path}.\n\n"
        f"Problem statement:\n{problem}\n\n"
        f"Failing tests that MUST pass after your fix:\n{ftp_str}\n\n"
        "Rules:\n"
        "- Make the minimum change required. Do not refactor unrelated code.\n"
        "- Do not edit test files.\n"
        "- Run the failing tests to verify your fix before finishing.\n"
    )

    if not no_graph:
        base += (
            "\nYou have access to the repo-insight MCP server with 22 code graph tools.\n"
            "Recommended workflow:\n"
            f"1. ingest_repository('{repo_path}')  — build the code graph first\n"
            "2. run_failing_tests_and_localize — find exact failing function from stack trace\n"
            "3. get_function_fingerprints — understand affected functions in ~30 tokens each\n"
            "4. get_source_code — read full source of 1-3 key functions\n"
            "5. Fix the bug, then run tests to verify\n"
        )

    return base


# All 23 Repo-Insight tool names — used to detect which tools Claude actually called
_REPO_INSIGHT_TOOLS = {
    "ingest_repository", "get_graph_summary", "get_architecture_diagram",
    "run_failing_tests_and_localize", "get_coverage_guided_blast_radius", "get_issue_context",
    "get_blast_radius", "get_impact_radius", "get_callers", "get_callees", "get_cross_module_callers",
    "get_source_code", "get_function_context", "get_file_interface",
    "semantic_search", "get_macro_architecture", "get_class_architecture", "get_module_readers",
    "get_function_fingerprints", "get_function_skeletons", "store_behavior_labels",
    "analyze_git_diff", "analyze_edit_impact",
}


def _parse_tool_calls(text: str) -> list[str]:
    """Extract Repo-Insight tool names that appear in Claude Code's output."""
    found = []
    for tool in _REPO_INSIGHT_TOOLS:
        if tool in text:
            found.append(tool)
    return sorted(found)


def _estimate_files_read(text: str) -> int:
    """Count distinct file paths read — proxy for 'how much context was needed'."""
    # Claude Code prints file reads as: Reading file_path or cat file_path
    patterns = [
        re.compile(r'Reading\s+([^\s]+\.py)'),
        re.compile(r'cat\s+([^\s]+\.py)'),
        re.compile(r'open\([\'"]([^\'"]+\.py)'),
    ]
    files: set[str] = set()
    for p in patterns:
        files.update(p.findall(text))
    return len(files)


# ---------------------------------------------------------------------------
# Token parser (Claude Code output format)
# ---------------------------------------------------------------------------

_TOKEN_PATTERNS = [
    # "↑ 12,345 ↓ 678" (Claude Code compact format)
    re.compile(r"↑\s*([\d,]+)\s*↓\s*([\d,]+)"),
    # "Input tokens: 12345, Output tokens: 678"
    re.compile(r"[Ii]nput tokens?:\s*([\d,]+).*?[Oo]utput tokens?:\s*([\d,]+)"),
    # "Tokens: 12345 input, 678 output"
    re.compile(r"[Tt]okens?:\s*([\d,]+)\s+input.*?([\d,]+)\s+output"),
    # "prompt_tokens: 12345, completion_tokens: 678"
    re.compile(r"prompt_tokens.*?([\d,]+).*?completion_tokens.*?([\d,]+)"),
]


def _parse_tokens(text: str) -> tuple[int, int]:
    """Extract (input_tokens, output_tokens) from Claude Code output. Returns (0, 0) on failure."""
    for pattern in _TOKEN_PATTERNS:
        m = pattern.search(text)
        if m:
            try:
                inp = int(m.group(1).replace(",", ""))
                out = int(m.group(2).replace(",", ""))
                return inp, out
            except (IndexError, ValueError):
                continue
    return 0, 0


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
        "no_graph": no_graph,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "agent": "claude-code-cli",
        # Metrics for picking Bob demo instances
        "tools_called": [],          # which Repo-Insight tools were actually invoked
        "files_read_estimate": 0,    # proxy for context consumed (fewer = more efficient)
        "agent_output_log": None,    # path to full per-instance output log
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

        _write_mcp_config(repo_dir, graph_name, no_graph)

        prompt = _build_prompt(instance, repo_dir, graph_name, no_graph)
        prompt_file = tmp_root / "prompt.txt"
        prompt_file.write_text(prompt)

        condition = "BASELINE" if no_graph else "GRAPH"
        logger.info("[%s] Running Claude Code (%s)", instance_id, condition)

        env = {
            **os.environ,
            "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", "fake"),
            # Force all Claude Code model tiers to our vLLM alias
            "ANTHROPIC_DEFAULT_OPUS_MODEL":   CLAUDE_MODEL,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": CLAUDE_MODEL,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL":  CLAUDE_MODEL,
        }
        if "ANTHROPIC_BASE_URL" not in env:
            logger.warning("[%s] ANTHROPIC_BASE_URL not set — Claude Code will call real Anthropic API", instance_id)

        proc = subprocess.run(
            ["claude", "--print", "--dangerously-skip-permissions",
             "--model", CLAUDE_MODEL, "-p", prompt],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=AGENT_TIMEOUT,
            env=env,
        )

        combined_output = proc.stdout + proc.stderr

        # Save full agent output to per-instance log file
        log_dir = output_dir / "agent_logs"
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / f"{instance_id}.log"
        log_path.write_text(
            f"=== Instance: {instance_id} | Condition: {'BASELINE' if no_graph else 'GRAPH'} ===\n"
            f"=== Duration: {round(time.monotonic()-t0,1)}s ===\n\n"
            f"--- STDOUT ---\n{proc.stdout}\n\n"
            f"--- STDERR ---\n{proc.stderr}\n"
        )
        result["agent_output_log"] = str(log_path)

        inp_tok, out_tok = _parse_tokens(combined_output)
        tools_called = _parse_tool_calls(combined_output) if not no_graph else []
        files_read   = _estimate_files_read(combined_output)

        result.update(
            input_tokens=inp_tok,
            output_tokens=out_tok,
            total_tokens=inp_tok + out_tok,
            tools_called=tools_called,
            files_read_estimate=files_read,
        )

        result["duration_s"] = round(time.monotonic() - t0, 1)

        if proc.returncode != 0 and not proc.stdout.strip():
            result["status"] = "error"
            result["error"] = (proc.stderr or "claude exited non-zero")[:500]
            return result

        patch = _capture_diff(repo_dir)
        result["patch"] = patch

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

    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["error"] = f"Claude Code timed out after {AGENT_TIMEOUT}s"
        result["duration_s"] = round(time.monotonic() - t0, 1)
    except FileNotFoundError:
        result["status"] = "error"
        result["error"] = "claude CLI not found — run: npm install -g @anthropic-ai/claude-code"
        result["duration_s"] = round(time.monotonic() - t0, 1)
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        result["duration_s"] = round(time.monotonic() - t0, 1)
        logger.error("[%s] Failed: %s", instance_id, e, exc_info=True)
    finally:
        if tmp_root and tmp_root.exists():
            shutil.rmtree(tmp_root, ignore_errors=True)

    logger.info("[%s] %s  %.1fs  tokens=%d  patch=%d chars",
                instance_id, result["status"], result["duration_s"],
                result["total_tokens"], len(result["patch"]))
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
    gname = f"{GRAPH_NAME}_ccli_w{worker_id}"
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

def _print_summary(results: list[dict], output_dir: Path) -> None:
    counts: dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    total    = len(results)
    patched  = counts.get("patched", 0)
    skipped  = counts.get("skipped", 0)
    empty    = counts.get("empty_diff", 0)
    errors   = counts.get("error", 0) + counts.get("syntax_error", 0) + counts.get("timeout", 0)
    non_skip = [r for r in results if r["status"] != "skipped"]
    avg_dur  = sum(r["duration_s"] for r in non_skip) / max(len(non_skip), 1)
    no_graph = results[0].get("no_graph", False) if results else False

    resolved  = [r for r in results if r["status"] == "patched"]
    attempted = [r for r in results if r["status"] != "skipped"]
    avg_tok_r = sum(r["total_tokens"] for r in resolved)  / max(len(resolved), 1)
    avg_tok_a = sum(r["total_tokens"] for r in attempted) / max(len(attempted), 1)

    condition = "BASELINE (no graph)" if no_graph else "GRAPH-ENHANCED (Repo-Insight)"

    print(f"\n{'='*60}")
    print(f"  Claude Code CLI — SWE-bench Verified Mini")
    print(f"  Condition: {condition}")
    print(f"{'='*60}")
    print(f"Dataset:  {SWEBENCH_DATASET}")
    print(f"Agent:    Claude Code CLI")
    print(f"")
    print(f"Resolution rate:    {patched}/{total}  ({100*patched/max(total,1):.1f}%)")
    print(f"Patched:            {patched}")
    print(f"Empty diff:         {empty}")
    print(f"Errors / timeouts:  {errors}")
    print(f"Skipped:            {skipped}")
    print(f"")
    if avg_tok_r > 0:
        print(f"Avg tokens/resolved:  {avg_tok_r:,.0f}")
        print(f"Avg tokens/attempted: {avg_tok_a:,.0f}")
    else:
        print(f"Token data: not available (upgrade Claude Code or check output format)")
    print(f"Avg time/instance:  {avg_dur:.0f}s")

    if not no_graph:
        # Tool usage breakdown — shows which tools Claude actually called
        from collections import Counter
        tool_counter: Counter = Counter()
        instances_with_tools = 0
        for r in attempted:
            tc = r.get("tools_called", [])
            if tc:
                instances_with_tools += 1
                tool_counter.update(tc)
        print(f"\nMCP tool adoption: {instances_with_tools}/{len(attempted)} instances called ≥1 tool")
        if tool_counter:
            print("Top tools used:")
            for tool, count in tool_counter.most_common(8):
                print(f"  {tool}: {count}×")

    print(f"\nPer-instance logs: {output_dir}/agent_logs/")

    # Write predictions.jsonl for official SWE-bench evaluator
    pred_path = output_dir / "predictions.jsonl"
    cond_tag  = "ccli-baseline" if no_graph else "ccli-graph"
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
                "model_name_or_path": cond_tag,
            }) + "\n")

    (output_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nPredictions: {pred_path}")
    print(f"Results dir: {output_dir}")
    print("\nNext: run official SWE-bench evaluator on predictions.jsonl")


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
            logging.FileHandler(log_dir / f"ccli_{ts}.log"),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SWE-bench Verified Mini — Claude Code CLI agent + Repo-Insight MCP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Max instances to run (default: all 50)")
    parser.add_argument("--instances", type=str, default=None,
                        help="Comma-separated instance IDs")
    parser.add_argument("--output-dir", default="./results/ccli_graph",
                        help="Directory for patches, logs, predictions.jsonl")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip instances that already have a .patch file")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel worker processes (default: 1)")
    parser.add_argument("--no-graph", action="store_true",
                        help="Baseline: disable Repo-Insight tools (no .claude/mcp.json servers)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(output_dir / "logs")

    # Verify claude CLI is available
    if not shutil.which("claude"):
        print("ERROR: claude CLI not found.")
        print("Install with: npm install -g @anthropic-ai/claude-code")
        sys.exit(1)

    no_graph = args.no_graph
    condition = "BASELINE (no graph)" if no_graph else "GRAPH-ENHANCED (Repo-Insight)"

    logger.info("Agent:      Claude Code CLI")
    logger.info("Condition:  %s", condition)
    logger.info("MCP server: %s", str(MCP_SERVER_PATH))
    logger.info("Workers:    %d", args.workers)

    if not os.environ.get("ANTHROPIC_BASE_URL"):
        logger.warning("ANTHROPIC_BASE_URL not set — Claude Code will use real Anthropic API (costs money)")
    else:
        logger.info("ANTHROPIC_BASE_URL: %s", os.environ["ANTHROPIC_BASE_URL"])

    logger.info("Loading %s ...", SWEBENCH_DATASET)
    ds = load_dataset(SWEBENCH_DATASET, split="test")
    dataset = list(ds)

    if args.instances:
        ids = {i.strip() for i in args.instances.split(",")}
        dataset = [d for d in dataset if d["instance_id"] in ids]
    if args.limit:
        dataset = dataset[: args.limit]

    logger.info("Running %d instances", len(dataset))

    n = max(1, args.workers)
    if n == 1:
        results = _run_sequential(dataset, output_dir, args.skip_existing, no_graph=no_graph)
    else:
        results = _run_parallel(dataset, output_dir, args.skip_existing, n, no_graph=no_graph)

    _print_summary(results, output_dir)


if __name__ == "__main__":
    main()
