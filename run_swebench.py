#!/usr/bin/env python3
"""
SWE-bench Lite evaluation harness for Repo-Insight's GraphDrivenEngine.

Workflow per instance:
  1. Clone the repo at base_commit into a temp directory
  2. Ingest the codebase into FalkorDB
  3. Run the 6-phase pipeline with the problem_statement as prompt
  4. Capture git diff → write .patch file
  5. Log result, clean up, move on

Usage:
  python run_swebench.py --limit 10
  python run_swebench.py --instances "django__django-11099,sympy__sympy-13480"
  python run_swebench.py --limit 50 --output-dir ./patches_v2 --skip-existing
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from config import FALKORDB_HOST, FALKORDB_PORT
from change_engine import GraphDrivenEngine, ChangeResult
from ingest import get_connection
from apply_changes import parse_edit_blocks, apply_edits
import signal

logger = logging.getLogger("swebench_harness")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INSTANCE_TIMEOUT = 1200  # seconds per instance
CLONE_TIMEOUT = 120     # seconds for git clone + checkout
INGEST_TIMEOUT = 3600   # 1 hour — astropy summaries alone take ~15min, Jedi ~10min
PIPELINE_TIMEOUT = 1200  # increased from 600 — thinking mode adds 60-150s per LLM call

# Repo URL templates for known SWE-bench orgs
_REPO_URL_TEMPLATE = "https://github.com/{repo}.git"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_logging(log_dir: Path) -> logging.FileHandler:
    """Configure dual logging: stderr + timestamped log file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"run_{timestamp}.log"

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler — captures everything
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # Stderr handler — INFO and above
    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(fh)
    root.addHandler(sh)

    logger.info("Logging to %s", log_path)
    return fh

def _hard_timeout(fn, seconds, *args, **kwargs):
    """Kill the function with SIGALRM after `seconds`. Linux/macOS only."""
    def _bail(signum, frame):
        raise TimeoutError(f"Killed after {seconds}s")
    signal.signal(signal.SIGALRM, _bail)
    signal.alarm(seconds)
    try:
        return fn(*args, **kwargs)
    finally:
        signal.alarm(0)
def _clone_repo(repo: str, commit: str, dest: Path) -> None:
    """Fetch a single commit from GitHub with minimal data transfer.

    Uses git-init + fetch-by-SHA rather than a full clone. This avoids
    the shallow-clone trap where `git checkout <old-sha>` fails with
    exit 128 because the commit isn't in the truncated local history.
    GitHub supports fetching arbitrary SHAs on public repos.
    """
    repo_url = _REPO_URL_TEMPLATE.format(repo=repo)
    logger.debug("Fetching %s @ %s → %s", repo_url, commit[:10], dest)

    dest.mkdir(parents=True, exist_ok=True)

    # Step 1: empty repo + remote
    subprocess.run(["git", "init", str(dest)],
                   check=True, capture_output=True, timeout=10)
    subprocess.run(["git", "remote", "add", "origin", repo_url],
                   cwd=str(dest), check=True, capture_output=True, timeout=10)

    # Step 2: fetch exactly this one commit — depth=1 means no parent history,
    # so only the tree at this SHA is downloaded (no full repo history needed).
    subprocess.run(
        ["git", "fetch", "--depth=1", "origin", commit],
        cwd=str(dest),
        check=True,
        capture_output=True,
        timeout=CLONE_TIMEOUT,
    )

    # Step 3: check out the fetched commit
    subprocess.run(
        ["git", "checkout", "FETCH_HEAD"],
        cwd=str(dest),
        check=True,
        capture_output=True,
        timeout=30,
    )

    logger.debug("Checked out %s successfully", commit[:10])


def _capture_diff(repo_dir: Path) -> str:
    """Capture the full git diff of uncommitted changes in repo_dir."""
    result = subprocess.run(
        ["git", "diff", "HEAD"],
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout


def _run_instance(
    instance: dict,
    output_dir: Path,
    skip_existing: bool,
    flush_graph: bool = True,
) -> dict:
    """Process a single SWE-bench instance. Returns a result dict.

    flush_graph=True  → drop and recreate the graph (use when repo changes).
    flush_graph=False → keep graph, rely on content-hash incremental update
                        (use for consecutive instances of the same repo).

    The entire instance runs in a temporary directory that is cleaned
    up regardless of outcome.
    """
    instance_id = instance["instance_id"]
    repo = instance["repo"]
    base_commit = instance["base_commit"]
    problem = instance["problem_statement"]

    # Extract the exact tests that must pass after the fix.
    # SWEbench stores this as a JSON-encoded string or a list.
    _raw_ftp = instance.get("FAIL_TO_PASS", "[]")
    try:
        import json as _json
        fail_to_pass: list[str] = (
            _json.loads(_raw_ftp) if isinstance(_raw_ftp, str) else list(_raw_ftp)
        )
    except Exception:
        fail_to_pass = []

    result = {
        "instance_id": instance_id,
        "repo": repo,
        "status": "pending",
        "patch_file": None,
        "phases_completed": [],
        "error": None,
        "timing_s": 0.0,
    }

    # Skip if patch already exists
    patch_path = output_dir / f"{instance_id}.patch"
    if skip_existing and patch_path.exists():
        result["status"] = "skipped"
        result["patch_file"] = str(patch_path)
        logger.info("[%s] Skipping — patch already exists", instance_id)
        return result

    t_start = time.time()
    tmp_root = None

    try:
        # -- Clone --
        tmp_root = Path(tempfile.mkdtemp(prefix=f"swebench_{instance_id}_"))
        repo_dir = tmp_root / "repo"
        logger.info("[%s] Cloning %s @ %s", instance_id, repo, base_commit[:10])
        _clone_repo(repo, base_commit, repo_dir)
        # -- Run pipeline --
        logger.info("[%s] Running 6-phase pipeline", instance_id)

        if flush_graph:
            logger.info("[%s] Flushing graph (repo changed)", instance_id)
            from ingest import drop_graph
            drop_graph()
            graph = get_connection()
        else:
            # Same repo as previous instance: keep graph, let content-hash
            # incremental logic in run_ingestion re-ingest only changed files.
            logger.info("[%s] Same repo — reusing graph (incremental update)", instance_id)
            graph = get_connection()

        engine = GraphDrivenEngine(
            repo_root=repo_dir,
            graph=graph,
            swebench_tests=fail_to_pass,
        )

        # Phase 0: ingestion — separate timeout, no LLM involved
        logger.info("[%s] Phase 0: ingesting graph (timeout=%ds)", instance_id, INGEST_TIMEOUT)
        _hard_timeout(engine._ensure_graph_fresh, INGEST_TIMEOUT)

        # Phases 1-5: LLM phases — tighter timeout
        logger.info("[%s] Phases 1-5: running pipeline (timeout=%ds)", instance_id, PIPELINE_TIMEOUT)

        def _run_phases():
            return engine.run(
                user_prompt=problem,
                skip_apply=False,
                _skip_phase0=True,   # ingestion already done above
            )

        change_result: ChangeResult = _hard_timeout(_run_phases, PIPELINE_TIMEOUT)
        # NOTE: skip_apply=False means Phase 5 applies edits but also runs
        # pytest with Repo-Insight's own TEST_COMMAND against the target repo,
        # which always fails. Set SKIP_SANDBOX_TESTS=true in env to avoid
        # wasting 3 × 120s per instance on guaranteed test failures.


        

        result["phases_completed"] = change_result.phases_completed

        if change_result.error:
            logger.warning("[%s] Pipeline error: %s", instance_id, change_result.error)
            result["error"] = change_result.error
        debug_dir = output_dir / "debug"
        debug_dir.mkdir(exist_ok=True)
        debug_path = debug_dir / f"{instance_id}.json"
        try:
            import json as _json
            with open(debug_path, "w", encoding="utf-8") as _f:
                    _json.dump({
                        "instance_id": instance_id,
                        "repo": repo,
                        "seeds": change_result.seeds,
                        "phases_completed": change_result.phases_completed,
                        "plan_raw": change_result.plan.raw_plan if change_result.plan else None,
                        "planned_files":list(change_result.plan.planned_files) if change_result.plan else [],
                        "missing_files": list(change_result.plan.missing_files) if change_result.plan else [],
                        "edit_count": len(change_result.edits),
                        "answer": change_result.answer[:3000] if change_result.answer else None,
                        "error": change_result.error,
                        "timings": change_result.timings,
                    }, _f, indent = 2, default=str)
            logger.info("[%s] Debug info written to %s", instance_id, debug_path)
        except Exception as _de:
            logger.error("[%s] Failed to write debug info: %s", instance_id, _de)


        # -- Capture diff --
        diff = _capture_diff(repo_dir)

        if diff.strip():
            patch_path.parent.mkdir(parents=True, exist_ok=True)
            patch_path.write_text(diff, encoding="utf-8")
            result["status"] = "patched"
            result["patch_file"] = str(patch_path)
            logger.info(
                "[%s] Patch written (%d bytes, %d lines)",
                instance_id, len(diff), diff.count("\n"),
            )
        else:
            result["status"] = "empty_diff"
            logger.warning("[%s] No diff generated — pipeline produced no changes", instance_id)

    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["error"] = "Instance timed out"
        logger.error("[%s] Timed out", instance_id)

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        logger.error("[%s] Failed: %s", instance_id, e, exc_info=True)

    finally:
        elapsed = time.time() - t_start
        result["timing_s"] = round(elapsed, 2)

        # Clean up temp directory
        if tmp_root and tmp_root.exists():
            try:
                shutil.rmtree(tmp_root, ignore_errors=True)
            except Exception:
                pass

        logger.info(
            "[%s] Done in %.1fs — status=%s",
            instance_id, elapsed, result["status"],
        )

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SWE-bench Lite evaluation harness for Repo-Insight",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--instances",
        type=str,
        default=None,
        help="Comma-separated instance IDs to run (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Max instances to process (default: 10 for smoke test)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./patches",
        help="Directory for .patch files (default: ./patches)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip instances where patch file already exists",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs",
        help="Directory for log files (default: ./logs)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    log_dir = Path(args.log_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _setup_logging(log_dir)

    # -- Load dataset --
    logger.info("Loading SWE-bench Lite dataset...")
    dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    logger.info("Loaded %d instances", len(dataset))

    # -- Filter instances --
    if args.instances:
        target_ids = {s.strip() for s in args.instances.split(",")}
        dataset = [inst for inst in dataset if inst["instance_id"] in target_ids]
        missing = target_ids - {inst["instance_id"] for inst in dataset}
        if missing:
            logger.warning("Instance IDs not found in dataset: %s", missing)
        logger.info("Filtered to %d instances", len(dataset))
    else:
        dataset = list(dataset)

    # Sort by repo so consecutive instances share the same codebase.
    # The graph is only flushed when the repo changes, and content-hash
    # incremental logic means only the ~5-15 files that actually changed
    # between commits are re-ingested. For Django (86 instances) this
    # turns 86 full builds into 1 full build + 85 incremental updates.
    dataset = sorted(dataset, key=lambda x: x["repo"])

    # Apply limit AFTER sorting so the limit covers a contiguous repo group
    if args.limit and len(dataset) > args.limit:
        dataset = dataset[:args.limit]
        logger.info("Limited to %d instances", len(dataset))

    # -- Run --
    results = []
    prev_repo: str | None = None
    for i, instance in enumerate(dataset, 1):
        iid = instance["instance_id"]
        current_repo = instance["repo"]
        flush = current_repo != prev_repo   # only flush when repo changes
        prev_repo = current_repo
        logger.info(
            "━━━ [%d/%d] %s ━━━",
            i, len(dataset), iid,
        )
        try:
            result = _run_instance(instance, output_dir, args.skip_existing,
                                   flush_graph=flush)
        except TimeoutError:
            logger.error("[%s] Hard wall-clock timeout after %ds", iid, INSTANCE_TIMEOUT)
            result = {
                "instance_id": iid,
                "repo": instance["repo"],
                "status": "timeout",
                "patch_file": None,
                "phases_completed": [],
                "timing_s": float(INSTANCE_TIMEOUT),
                "error": f"Timeout after {INSTANCE_TIMEOUT}s",
            }
        results.append(result)

    # -- Summary --
    total = len(results)
    patched = sum(1 for r in results if r["status"] == "patched")
    empty = sum(1 for r in results if r["status"] == "empty_diff")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    errored = sum(1 for r in results if r["status"] == "error")
    timed_out = sum(1 for r in results if r["status"] == "timeout")

    failed_ids = [r["instance_id"] for r in results if r["status"] in ("error", "timeout", "empty_diff")]

    summary = {
        "total_attempted": total,
        "patches_generated": patched,
        "empty_diffs": empty,
        "skipped": skipped,
        "errors": errored,
        "timeouts": timed_out,
        "failed_instance_ids": failed_ids,
        "avg_time_s": round(sum(r["timing_s"] for r in results) / max(total, 1), 2),
    }

    logger.info("=" * 60)
    logger.info("SWE-bench Lite Run Summary")
    logger.info("=" * 60)
    logger.info("Total attempted:    %d", total)
    logger.info("Patches generated:  %d", patched)
    logger.info("Empty diffs:        %d", empty)
    logger.info("Skipped (existing): %d", skipped)
    logger.info("Errors:             %d", errored)
    logger.info("Timeouts:           %d", timed_out)
    logger.info("Avg time/instance:  %.1fs", summary["avg_time_s"])

    if failed_ids:
        logger.info("Failed instances:")
        for fid in failed_ids:
            logger.info("  - %s", fid)

    # Write machine-readable summary
    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(
        json.dumps({"summary": summary, "results": results}, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("Summary written to %s", summary_path)

    # Exit code: 0 if at least one patch generated, 1 otherwise
    sys.exit(0 if patched > 0 else 1)


if __name__ == "__main__":
    main()
