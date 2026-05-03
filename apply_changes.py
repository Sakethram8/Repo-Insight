# apply_changes.py
"""
Parse SEARCH/REPLACE edit blocks from LLM output, apply them to a sandboxed
copy of the repository, and run tests to verify correctness.
"""

import difflib
import logging
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EditBlock:
    """A single SEARCH/REPLACE edit."""
    file_path: str
    search_text: str
    replace_text: str


@dataclass
class FileApplyResult:
    """Result of applying edits to a single file."""
    file_path: str
    success: bool
    error: Optional[str] = None
    match_method: str = "exact"  # "exact" or "fuzzy"


@dataclass
class ApplyResult:
    """Aggregate result of applying all edits."""
    file_results: list[FileApplyResult] = field(default_factory=list)
    total_edits: int = 0
    successful_edits: int = 0
    failed_edits: int = 0
    failed_edits_detail: list[dict] = field(default_factory=list)

    @property
    def all_succeeded(self) -> bool:
        return self.failed_edits == 0


@dataclass
class TestResult:
    """Result of running pytest in the sandbox."""
    passed: int = 0
    failed: int = 0
    errors: int = 0
    exit_code: int = -1
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False

    @property
    def all_passed(self) -> bool:
        return self.exit_code == 0 and self.failed == 0 and self.errors == 0


# ---------------------------------------------------------------------------
# Edit block parsing
# ---------------------------------------------------------------------------

# Pattern matches:
# FILE: path/to/file.py
# <<<<<<< SEARCH
# ...existing code...
# =======
# ...replacement code...
# >>>>>>> REPLACE
_EDIT_BLOCK_PATTERN = re.compile(
    r"FILE:\s*(.+?)\s*\n"
    r"<<<<<<< SEARCH\n"
    r"(.*?)\n"
    r"=======\n"
    r"(.*?)\n"
    r">>>>>>> REPLACE",
    re.DOTALL,
)


def parse_edit_blocks(llm_output: str) -> list[EditBlock]:
    """Parse SEARCH/REPLACE blocks from LLM output.

    Handles multiple blocks per file and minor formatting variations.
    """
    blocks = []
    for match in _EDIT_BLOCK_PATTERN.finditer(llm_output):
        file_path = match.group(1).strip().strip("`").strip("'").strip('"')
        search_text = match.group(2)
        replace_text = match.group(3)
        blocks.append(EditBlock(
            file_path=file_path,
            search_text=search_text,
            replace_text=replace_text,
        ))

    if not blocks:
        logger.warning("No SEARCH/REPLACE blocks found in LLM output")

    return blocks


# ---------------------------------------------------------------------------
# Edit application
# ---------------------------------------------------------------------------

def apply_edits(edit_blocks: list[EditBlock], target_dir: Path) -> ApplyResult:
    """Apply all edit blocks to files in the target directory.

    Args:
        edit_blocks: Parsed SEARCH/REPLACE blocks.
        target_dir: Root directory to apply edits to (usually the sandbox).

    Returns:
        ApplyResult with per-file success/failure details.
    """
    result = ApplyResult(total_edits=len(edit_blocks))
    target_dir = target_dir.resolve()

    for edit in edit_blocks:
        file_path = (target_dir / edit.file_path).resolve()
        if not str(file_path).startswith(str(target_dir) + "/"):
            result.file_results.append(FileApplyResult(
                file_path=edit.file_path, success=False,
                error=f"Security: path escapes sandbox: {edit.file_path}",
            ))
            result.failed_edits += 1
            result.failed_edits_detail.append({
                "file_path": edit.file_path,
                "reason": "Path escapes sandbox",
            })
            continue

        if not file_path.exists():
            result.file_results.append(FileApplyResult(
                file_path=edit.file_path,
                success=False,
                error=f"File not found: {edit.file_path}",
            ))
            result.failed_edits += 1
            result.failed_edits_detail.append({
                "file_path": edit.file_path,
                "reason": "File not found",
            })
            continue

        try:
            content = file_path.read_text(encoding="utf-8")
            
            # 1. Try exact match first
            new_content = content.replace(edit.search_text, edit.replace_text, 1)
            method = "exact"
            
            # 2. If exact match fails, try normalized match
            if new_content == content:
                orig_lines = content.splitlines(keepends=True)
                search_lines = edit.search_text.splitlines()
                
                norm_search = [line.rstrip() for line in search_lines]
                norm_content = [line.rstrip() for line in content.splitlines()]
                
                search_len = len(norm_search)
                match_idx = -1
                
                if search_len > 0:
                    for i in range(len(norm_content) - search_len + 1):
                        if norm_content[i:i + search_len] == norm_search:
                            match_idx = i
                            break
                
                if match_idx >= 0:
                    before = "".join(orig_lines[:match_idx])
                    after = "".join(orig_lines[match_idx + search_len:])
                    
                    replaced_text_orig = "".join(orig_lines[match_idx:match_idx + search_len])
                    replace_text = edit.replace_text
                    if replaced_text_orig.endswith("\n") and not replace_text.endswith("\n"):
                        replace_text += "\n"
                        
                    new_content = before + replace_text + after
                    method = "normalized"
                else:
                    logger.warning(
                        "Edit block failed for %s — search text not found. First 80 chars of search: %s",
                        edit.file_path, edit.search_text[:80]
                    )
                    result.file_results.append(FileApplyResult(
                        file_path=edit.file_path,
                        success=False,
                        error="search text not found",
                    ))
                    result.failed_edits += 1
                    result.failed_edits_detail.append({
                        "file_path": edit.file_path,
                        "reason": "search text not found",
                    })
                    continue

            file_path.write_text(new_content, encoding="utf-8")
            result.file_results.append(FileApplyResult(
                file_path=edit.file_path,
                success=True,
                match_method=method,
            ))
            result.successful_edits += 1
            logger.info("Applied edit to %s (method: %s)", edit.file_path, method)

        except Exception as e:
            result.file_results.append(FileApplyResult(
                file_path=edit.file_path,
                success=False,
                error=f"Unexpected error: {e}",
            ))
            result.failed_edits += 1
            result.failed_edits_detail.append({
                "file_path": edit.file_path,
                "reason": f"Unexpected error: {e}",
            })
            logger.error("Unexpected error applying edit to %s: %s", edit.file_path, e)

    return result


# ---------------------------------------------------------------------------
# Sandbox management
# ---------------------------------------------------------------------------

def create_sandbox(repo_root: Path) -> Path:
    """Create a temporary copy of the repository for safe edit testing.

    Returns the path to the sandbox directory.
    The caller is responsible for cleanup via `cleanup_sandbox()`.
    """
    import os
    from config import SKIP_DIRS

    sandbox_dir = Path(tempfile.mkdtemp(prefix="repo_insight_sandbox_"))
    target = sandbox_dir / "repo"

    # Copy repo, ignoring heavy/irrelevant directories
    ignore = shutil.ignore_patterns(
        *SKIP_DIRS, "*.pyc", ".mypy_cache", ".pytest_cache", "*.egg-info",
    )
    
    try:
        # Attempt to use hardlinks for near-instantaneous sandbox creation
        shutil.copytree(repo_root, target, ignore=ignore, copy_function=os.link, dirs_exist_ok=True)
    except OSError:
        # Fallback to standard copy if hardlinks fail (e.g., across filesystems)
        shutil.copytree(repo_root, target, ignore=ignore, dirs_exist_ok=True)

    logger.info("Created sandbox at %s", target)
    return target


def cleanup_sandbox(sandbox_path: Path) -> None:
    """Remove a sandbox directory."""
    try:
        # sandbox_path is repo_root/repo, parent is the temp dir
        parent = sandbox_path.parent
        if parent.name.startswith("repo_insight_sandbox_"):
            shutil.rmtree(parent, ignore_errors=True)
        else:
            shutil.rmtree(sandbox_path, ignore_errors=True)
    except Exception as e:
        logger.warning("Failed to cleanup sandbox: %s", e)


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_tests(
    sandbox_path: Path,
    timeout: int = 120,
    test_command: Optional[list[str]] = None,
) -> TestResult:
    """Run pytest in the sandbox directory.

    Args:
        sandbox_path: Path to the sandboxed repo.
        timeout: Maximum seconds to wait for tests.
        test_command: Custom test command. Defaults to pytest with minimal output.

    Returns:
        TestResult with pass/fail counts and output.
    """
    if test_command is None:
        import shlex
        from config import TEST_COMMAND
        test_command = shlex.split(TEST_COMMAND)

    try:
        proc = subprocess.run(
            test_command,
            cwd=str(sandbox_path),
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        result = TestResult(
            exit_code=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )

        # Parse pytest output for pass/fail counts
        # Look for "X passed, Y failed, Z errors" pattern
        summary_match = re.search(
            r"(\d+) passed", proc.stdout + proc.stderr,
        )
        if summary_match:
            result.passed = int(summary_match.group(1))

        fail_match = re.search(
            r"(\d+) failed", proc.stdout + proc.stderr,
        )
        if fail_match:
            result.failed = int(fail_match.group(1))

        error_match = re.search(
            r"(\d+) error", proc.stdout + proc.stderr,
        )
        if error_match:
            result.errors = int(error_match.group(1))

        return result

    except subprocess.TimeoutExpired:
        logger.error("Tests timed out after %ds", timeout)
        return TestResult(
            exit_code=-1,
            timed_out=True,
            stdout="",
            stderr=f"Tests timed out after {timeout} seconds",
        )
    except Exception as e:
        logger.error("Failed to run tests: %s", e)
        return TestResult(
            exit_code=-1,
            stdout="",
            stderr=f"Failed to run tests: {e}",
        )


def apply_to_original(sandbox_path: Path, original_path: Path):
    """
    Copy changed files from sandbox back to original repo.
    Delegates to SandboxManager.apply_to_original().
    Import is deferred to avoid circular imports.
    """
    from sandbox import SandboxManager
    manager = SandboxManager(original_path)
    manager.sandbox_path = sandbox_path
    manager.sandbox_id = sandbox_path.name
    return manager.apply_to_original()
