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

_FUZZY_THRESHOLD = 0.85  # SequenceMatcher ratio for fuzzy matching


def _find_and_replace(
    content: str,
    search_text: str,
    replace_text: str,
) -> tuple[str, str]:
    """Find search_text in content and replace it.

    Returns (new_content, match_method).
    Raises ValueError if search_text cannot be found even with fuzzy matching.
    """
    # Try exact match first
    if search_text in content:
        return content.replace(search_text, replace_text, 1), "exact"

    # Try with normalized whitespace (strip trailing spaces per line)
    search_normalized = "\n".join(line.rstrip() for line in search_text.splitlines())
    content_normalized = "\n".join(line.rstrip() for line in content.splitlines())

    if search_normalized in content_normalized:
        # Find the position in normalized, apply to original
        idx = content_normalized.index(search_normalized)
        # Map back to original content by counting characters
        original_lines = content.splitlines(keepends=True)
        normalized_lines = content_normalized.splitlines(keepends=True)

        # Rebuild with replacement
        before = content_normalized[:idx]
        after = content_normalized[idx + len(search_normalized):]
        new_normalized = before + replace_text + after

        # Re-add original line endings
        return new_normalized, "whitespace_normalized"

    # Fuzzy matching: find the best matching region in the file
    search_lines = search_text.splitlines()
    content_lines = content.splitlines()

    if not search_lines:
        raise ValueError("Empty search text")

    best_ratio = 0.0
    best_start = 0
    best_end = 0
    window = len(search_lines)

    for i in range(len(content_lines) - window + 1):
        candidate = "\n".join(content_lines[i:i + window])
        ratio = difflib.SequenceMatcher(None, search_text, candidate).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_start = i
            best_end = i + window

    if best_ratio >= _FUZZY_THRESHOLD:
        new_lines = content_lines[:best_start] + replace_text.splitlines() + content_lines[best_end:]
        return "\n".join(new_lines), f"fuzzy({best_ratio:.2f})"

    raise ValueError(
        f"Could not find search text in file (best fuzzy match: {best_ratio:.2f}, "
        f"threshold: {_FUZZY_THRESHOLD}). First 80 chars of search: "
        f"'{search_text[:80]}...'"
    )


def apply_edits(edit_blocks: list[EditBlock], target_dir: Path) -> ApplyResult:
    """Apply all edit blocks to files in the target directory.

    Args:
        edit_blocks: Parsed SEARCH/REPLACE blocks.
        target_dir: Root directory to apply edits to (usually the sandbox).

    Returns:
        ApplyResult with per-file success/failure details.
    """
    result = ApplyResult(total_edits=len(edit_blocks))

    for block in edit_blocks:
        file_path = target_dir / block.file_path
        if not file_path.exists():
            result.file_results.append(FileApplyResult(
                file_path=block.file_path,
                success=False,
                error=f"File not found: {block.file_path}",
            ))
            result.failed_edits += 1
            continue

        try:
            content = file_path.read_text(encoding="utf-8")
            new_content, method = _find_and_replace(
                content, block.search_text, block.replace_text,
            )
            file_path.write_text(new_content, encoding="utf-8")
            result.file_results.append(FileApplyResult(
                file_path=block.file_path,
                success=True,
                match_method=method,
            ))
            result.successful_edits += 1
            logger.info("Applied edit to %s (method: %s)", block.file_path, method)
        except ValueError as e:
            result.file_results.append(FileApplyResult(
                file_path=block.file_path,
                success=False,
                error=str(e),
            ))
            result.failed_edits += 1
            logger.error("Failed to apply edit to %s: %s", block.file_path, e)
        except Exception as e:
            result.file_results.append(FileApplyResult(
                file_path=block.file_path,
                success=False,
                error=f"Unexpected error: {e}",
            ))
            result.failed_edits += 1
            logger.error("Unexpected error applying edit to %s: %s", block.file_path, e)

    return result


# ---------------------------------------------------------------------------
# Sandbox management
# ---------------------------------------------------------------------------

def create_sandbox(repo_root: Path) -> Path:
    """Create a temporary copy of the repository for safe edit testing.

    Returns the path to the sandbox directory.
    The caller is responsible for cleanup via `cleanup_sandbox()`.
    """
    sandbox_dir = Path(tempfile.mkdtemp(prefix="repo_insight_sandbox_"))
    target = sandbox_dir / "repo"

    # Copy repo, ignoring heavy/irrelevant directories
    ignore = shutil.ignore_patterns(
        ".git", "__pycache__", "*.pyc", ".venv", "venv",
        "node_modules", ".mypy_cache", ".pytest_cache",
        "dist", "build", "*.egg-info",
    )
    shutil.copytree(repo_root, target, ignore=ignore)
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
        test_command = [
            "python", "-m", "pytest", "tests/", "-v",
            "--tb=short", "-q",
            "-m", "not integration",  # Skip integration tests in sandbox
        ]

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
