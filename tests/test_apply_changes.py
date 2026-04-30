# tests/test_apply_changes.py
"""
Unit tests for apply_changes.py — edit block parsing, text matching,
sandbox management, and test runner.
No live infrastructure required.
"""

import pytest
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch
from apply_changes import (
    EditBlock, FileApplyResult, ApplyResult, TestResult,
    parse_edit_blocks, _find_and_replace, apply_edits,
    create_sandbox, cleanup_sandbox, run_tests,
)


# ---------------------------------------------------------------------------
# Edit block parsing
# ---------------------------------------------------------------------------

class TestParseEditBlocks:
    def test_single_block(self):
        llm_output = (
            "FILE: parser.py\n"
            "<<<<<<< SEARCH\n"
            "    name: str\n"
            "=======\n"
            "    name: str\n"
            "    decorators: list[str]\n"
            ">>>>>>> REPLACE\n"
        )
        blocks = parse_edit_blocks(llm_output)
        assert len(blocks) == 1
        assert blocks[0].file_path == "parser.py"
        assert "name: str" in blocks[0].search_text
        assert "decorators" in blocks[0].replace_text

    def test_multiple_blocks_same_file(self):
        llm_output = (
            "FILE: parser.py\n"
            "<<<<<<< SEARCH\n"
            "line1\n"
            "=======\n"
            "line1_new\n"
            ">>>>>>> REPLACE\n"
            "\n"
            "FILE: parser.py\n"
            "<<<<<<< SEARCH\n"
            "line2\n"
            "=======\n"
            "line2_new\n"
            ">>>>>>> REPLACE\n"
        )
        blocks = parse_edit_blocks(llm_output)
        assert len(blocks) == 2
        assert all(b.file_path == "parser.py" for b in blocks)

    def test_multiple_blocks_different_files(self):
        llm_output = (
            "FILE: a.py\n"
            "<<<<<<< SEARCH\n"
            "old_a\n"
            "=======\n"
            "new_a\n"
            ">>>>>>> REPLACE\n"
            "\n"
            "FILE: b.py\n"
            "<<<<<<< SEARCH\n"
            "old_b\n"
            "=======\n"
            "new_b\n"
            ">>>>>>> REPLACE\n"
        )
        blocks = parse_edit_blocks(llm_output)
        assert len(blocks) == 2
        assert {b.file_path for b in blocks} == {"a.py", "b.py"}

    def test_no_blocks_returns_empty(self):
        llm_output = "Here are the changes I recommend:\n1. Modify parser.py\n2. Done"
        blocks = parse_edit_blocks(llm_output)
        assert blocks == []

    def test_file_path_with_backticks_stripped(self):
        llm_output = (
            "FILE: `parser.py`\n"
            "<<<<<<< SEARCH\n"
            "old\n"
            "=======\n"
            "new\n"
            ">>>>>>> REPLACE\n"
        )
        blocks = parse_edit_blocks(llm_output)
        assert len(blocks) == 1
        assert blocks[0].file_path == "parser.py"

    def test_file_path_with_quotes_stripped(self):
        llm_output = (
            "FILE: \"parser.py\"\n"
            "<<<<<<< SEARCH\n"
            "old\n"
            "=======\n"
            "new\n"
            ">>>>>>> REPLACE\n"
        )
        blocks = parse_edit_blocks(llm_output)
        assert len(blocks) == 1
        assert blocks[0].file_path == "parser.py"

    def test_multiline_search_and_replace(self):
        llm_output = (
            "FILE: test.py\n"
            "<<<<<<< SEARCH\n"
            "def foo():\n"
            "    return 1\n"
            "=======\n"
            "def foo():\n"
            "    return 2\n"
            ">>>>>>> REPLACE\n"
        )
        blocks = parse_edit_blocks(llm_output)
        assert len(blocks) == 1
        assert "def foo():\n    return 1" in blocks[0].search_text
        assert "def foo():\n    return 2" in blocks[0].replace_text


# ---------------------------------------------------------------------------
# Find and replace
# ---------------------------------------------------------------------------

class TestFindAndReplace:
    def test_exact_match(self):
        content = "line1\nline2\nline3\n"
        new, method = _find_and_replace(content, "line2", "LINE2")
        assert method == "exact"
        assert "LINE2" in new
        assert "line2" not in new

    def test_whitespace_normalized_match(self):
        content = "line1  \nline2  \nline3\n"
        new, method = _find_and_replace(content, "line1\nline2", "LINE1\nLINE2")
        assert method == "whitespace_normalized"

    def test_fuzzy_match_above_threshold(self):
        content = "def foo(a, b):\n    return a + b\n\ndef bar():\n    pass\n"
        # Slightly different version (extra space)
        search = "def foo(a,  b):\n    return a + b"
        replace = "def foo(a, b, c):\n    return a + b + c"
        new, method = _find_and_replace(content, search, replace)
        assert "fuzzy" in method
        assert "a + b + c" in new

    def test_no_match_raises_valueerror(self):
        content = "completely different content here"
        with pytest.raises(ValueError, match="Could not find"):
            _find_and_replace(content, "nothing like this at all\nfor real", "replacement")

    def test_empty_search_returns_exact_match(self):
        """Empty search text matches immediately via 'in' check."""
        content = "some content"
        new, method = _find_and_replace(content, "", "replacement")
        assert method == "exact"


# ---------------------------------------------------------------------------
# Apply edits
# ---------------------------------------------------------------------------

class TestApplyEdits:
    def test_successful_single_edit(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("def foo():\n    return 1\n")

        edits = [EditBlock(file_path="test.py", search_text="return 1", replace_text="return 2")]
        result = apply_edits(edits, tmp_path)

        assert result.all_succeeded is True
        assert result.successful_edits == 1
        assert result.failed_edits == 0
        assert f.read_text() == "def foo():\n    return 2\n"

    def test_file_not_found(self, tmp_path):
        edits = [EditBlock(file_path="nonexistent.py", search_text="x", replace_text="y")]
        result = apply_edits(edits, tmp_path)

        assert result.all_succeeded is False
        assert result.failed_edits == 1
        assert "not found" in result.file_results[0].error.lower()

    def test_mixed_success_failure(self, tmp_path):
        good_file = tmp_path / "good.py"
        good_file.write_text("old_content\n")

        edits = [
            EditBlock(file_path="good.py", search_text="old_content", replace_text="new_content"),
            EditBlock(file_path="missing.py", search_text="x", replace_text="y"),
        ]
        result = apply_edits(edits, tmp_path)

        assert result.total_edits == 2
        assert result.successful_edits == 1
        assert result.failed_edits == 1
        assert result.all_succeeded is False

    def test_search_not_found_in_file(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("actual content\n")

        edits = [EditBlock(
            file_path="test.py",
            search_text="completely different text\nthat spans multiple lines",
            replace_text="replacement",
        )]
        result = apply_edits(edits, tmp_path)
        assert result.failed_edits == 1


# ---------------------------------------------------------------------------
# Sandbox management
# ---------------------------------------------------------------------------

class TestSandbox:
    def test_create_sandbox_creates_directory(self, tmp_path):
        # Create a mini repo
        src = tmp_path / "src_repo"
        src.mkdir()
        (src / "main.py").write_text("print('hello')")
        (src / "sub").mkdir()
        (src / "sub" / "mod.py").write_text("x = 1")

        sandbox = create_sandbox(src)
        try:
            assert sandbox.exists()
            assert (sandbox / "main.py").exists()
            assert (sandbox / "sub" / "mod.py").exists()
        finally:
            cleanup_sandbox(sandbox)

    def test_sandbox_ignores_git_and_venv(self, tmp_path):
        src = tmp_path / "src_repo"
        src.mkdir()
        (src / "main.py").write_text("code")
        (src / ".git").mkdir()
        (src / ".git" / "HEAD").write_text("ref")
        (src / "venv").mkdir()
        (src / "venv" / "bin").mkdir()
        (src / "venv" / "bin" / "python").write_text("#!/bin/sh")

        sandbox = create_sandbox(src)
        try:
            assert not (sandbox / ".git").exists()
            assert not (sandbox / "venv").exists()
        finally:
            cleanup_sandbox(sandbox)

    def test_cleanup_removes_directory(self, tmp_path):
        src = tmp_path / "src_repo"
        src.mkdir()
        (src / "f.py").write_text("x")

        sandbox = create_sandbox(src)
        assert sandbox.exists()
        cleanup_sandbox(sandbox)
        # Parent temp dir should be gone
        assert not sandbox.parent.exists() or not sandbox.exists()


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

class TestRunTests:
    def test_successful_tests(self, tmp_path):
        import sys
        # Create a minimal test file
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_ok.py").write_text("def test_pass(): assert True\n")

        result = run_tests(
            tmp_path,
            test_command=[sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        )
        assert result.exit_code == 0
        assert result.passed >= 1

    def test_failing_tests(self, tmp_path):
        import sys
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_fail.py").write_text("def test_fail(): assert False\n")

        result = run_tests(
            tmp_path,
            test_command=[sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        )
        assert result.exit_code != 0
        assert result.failed >= 1

    def test_timeout_handling(self, tmp_path):
        import sys
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_slow.py").write_text(
            "import time\ndef test_slow(): time.sleep(10)\n"
        )
        result = run_tests(
            tmp_path,
            timeout=2,
            test_command=[sys.executable, "-m", "pytest", "tests/", "-v"],
        )
        assert result.timed_out is True

    def test_all_passed_property(self):
        r = TestResult(exit_code=0, passed=5, failed=0, errors=0)
        assert r.all_passed is True

    def test_all_passed_false_with_failures(self):
        r = TestResult(exit_code=1, passed=3, failed=2, errors=0)
        assert r.all_passed is False

    def test_all_passed_false_with_errors(self):
        r = TestResult(exit_code=1, passed=3, failed=0, errors=1)
        assert r.all_passed is False
