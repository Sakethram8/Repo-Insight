# git_tools.py
"""
Git-aware impact analysis for Repo-Insight.

Given a git ref, computes which external callers break due to interface
changes (modified files) or total deletions (deleted files) in that commit.

The algorithm:
  1. Discover changed .py files via git diff
  2. Capture old function interfaces from the graph (pre-reingest)
  3. Capture callers of to-be-deleted functions before they are removed
  4. Re-ingest all changed files into the graph
  5. Diff old vs new interfaces → build changed_signatures list
  6. Feed changed_signatures to analyze_edit_impact → broken callers
  7. Return a unified impact report
"""

import logging
import subprocess
from pathlib import Path
from typing import Optional

import falkordb

from ingest import reingest_files
from tools import get_file_interface, analyze_edit_impact, get_cross_module_callers

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _git(repo_root: Path, *args: str) -> list[str]:
    """Run a git command in *repo_root*, return stdout lines (stripped)."""
    output = subprocess.check_output(
        ["git", "-C", str(repo_root)] + list(args),
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
    return output.splitlines() if output else []


def _get_changed_files(ref: str, repo_root: Path) -> tuple[list[str], list[str]]:
    """Return (modified_py_files, deleted_py_files) for the given ref vs its parent."""
    modified = [
        f for f in _git(repo_root, "diff", "--name-only", "--diff-filter=ACMR",
                        f"{ref}~1", ref)
        if f.endswith(".py")
    ]
    deleted = [
        f for f in _git(repo_root, "diff", "--name-only", "--diff-filter=D",
                        f"{ref}~1", ref)
        if f.endswith(".py")
    ]
    return modified, deleted


def _diff_signatures(
    old_iface: dict, new_iface: dict
) -> tuple[list[dict], list[str]]:
    """Compare two get_file_interface results.

    Returns:
        changed_signatures: list of {fqn, old_params, new_params, old_return, new_return}
            for functions whose observable interface changed.
        removed_fqns: FQNs present in old_iface but absent in new_iface.
    """
    old_fns = {f["fqn"]: f for f in old_iface.get("functions", [])}
    new_fns = {f["fqn"]: f for f in new_iface.get("functions", [])}

    changed: list[dict] = []
    for fqn, old_fn in old_fns.items():
        if fqn not in new_fns:
            continue  # removed — handled separately
        new_fn = new_fns[fqn]
        params_changed = old_fn.get("params", []) != new_fn.get("params", [])
        return_changed = (old_fn.get("return_annotation") or "") != (new_fn.get("return_annotation") or "")
        if params_changed or return_changed:
            changed.append({
                "fqn": fqn,
                "old_params": old_fn.get("params", []),
                "new_params": new_fn.get("params", []),
                "old_return": old_fn.get("return_annotation"),
                "new_return": new_fn.get("return_annotation"),
            })

    removed = [fqn for fqn in old_fns if fqn not in new_fns]
    return changed, removed


def _resolve_repo_root(
    repo_root_arg: Optional[str | Path],
    graph: falkordb.Graph,
) -> Path:
    """Resolve repo_root from arg → git → graph Meta node."""
    if repo_root_arg:
        return Path(repo_root_arg)
    try:
        lines = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip().splitlines()
        if lines:
            return Path(lines[0])
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    try:
        res = graph.query("MATCH (m:Meta {key: 'repo_root'}) RETURN m.value LIMIT 1")
        if res.result_set:
            return Path(res.result_set[0][0])
    except Exception:
        pass
    raise ValueError(
        "Cannot determine repo_root. "
        "Pass it explicitly or run from inside the git repository."
    )


def _capture_deleted_fn_breaks(deleted_files: list[str], graph: falkordb.Graph) -> list[dict]:
    """Before deleted files are removed from the graph, find their cross-module callers."""
    breaks: list[dict] = []
    for rel_path in deleted_files:
        iface = get_file_interface(rel_path, graph)
        for fn in iface.get("functions", []):
            callers_info = get_cross_module_callers(fn["fqn"], graph)
            callers = callers_info.get("callers", [])
            if callers:
                breaks.append({
                    "deleted_fqn": fn["fqn"],
                    "file_path": rel_path,
                    "broken_callers": callers,
                    "caller_count": len(callers),
                })
    return breaks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def git_diff_impact(
    ref: str = "HEAD",
    repo_root: Optional[str | Path] = None,
    graph: Optional[falkordb.Graph] = None,
) -> dict:
    """Compute which external callers break due to changes at *ref*.

    Args:
        ref:       Git ref to analyse (default: HEAD — most recent commit).
        repo_root: Repository root path. Auto-detected if omitted.
        graph:     FalkorDB graph. If None, a connection is created.

    Returns:
        {
            ref, changed_files, deleted_files,
            interface_breaks, reader_breaks, deleted_function_breaks,
            total_at_risk,
        }
    """
    if graph is None:
        from ingest import get_connection
        graph = get_connection()

    resolved_root = _resolve_repo_root(repo_root, graph)

    try:
        modified, deleted = _get_changed_files(ref, resolved_root)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.error("git diff failed for ref=%r: %s", ref, e)
        return {
            "ref": ref, "error": str(e),
            "changed_files": [], "deleted_files": [],
            "interface_breaks": [], "reader_breaks": [],
            "deleted_function_breaks": [], "total_at_risk": 0,
        }

    logger.info(
        "git_diff_impact ref=%r: %d modified, %d deleted Python files",
        ref, len(modified), len(deleted),
    )

    # Step 1 — capture callers of deleted functions BEFORE graph cleanup
    deleted_fn_breaks = _capture_deleted_fn_breaks(deleted, graph)

    # Step 2 — snapshot old signatures for modified files
    old_ifaces = {fp: get_file_interface(fp, graph) for fp in modified}

    # Step 3 — re-ingest all changed files (reingest handles missing files gracefully)
    all_changed = modified + deleted
    if all_changed:
        reingest_files(all_changed, graph, resolved_root)

    # Step 4 — get new signatures
    new_ifaces = {fp: get_file_interface(fp, graph) for fp in modified}

    # Step 5 — diff interfaces and collect breaking callers
    interface_breaks: list[dict] = []
    for fp in modified:
        changed_sigs, _ = _diff_signatures(old_ifaces[fp], new_ifaces[fp])
        if not changed_sigs:
            continue
        impact = analyze_edit_impact(fp, changed_sigs, graph)
        interface_breaks.extend(impact.get("impact", []))

    total_at_risk = (
        sum(e["caller_count"] for e in interface_breaks)
        + sum(e["caller_count"] for e in deleted_fn_breaks)
    )

    return {
        "ref": ref,
        "changed_files": modified,
        "deleted_files": deleted,
        "interface_breaks": interface_breaks,
        "reader_breaks": [],  # reserved: requires parsing old file content via git show
        "deleted_function_breaks": deleted_fn_breaks,
        "total_at_risk": total_at_risk,
    }
