# tests/test_ingest.py
"""
Unit tests for ingest.py.
Uses mocks for FalkorDB and LLM calls. Tests helpers and ingestion logic.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from ingest import (
    _file_to_module, extract_source_code,
)


class TestFileToModule:
    def test_simple_file(self):
        assert _file_to_module("parser.py") == "parser"

    def test_nested_path(self):
        assert _file_to_module("pkg/subpkg/module.py") == "pkg.subpkg.module"

    def test_windows_path(self):
        assert _file_to_module("pkg\\subpkg\\module.py") == "pkg.subpkg.module"

    def test_no_py_extension(self):
        assert _file_to_module("pkg.module") == "pkg.module"

    def test_init_file(self):
        assert _file_to_module("pkg/__init__.py") == "pkg.__init__"


class TestExtractSourceCode:
    def test_extracts_correct_lines(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("line1\nline2\nline3\nline4\nline5\n")
        result = extract_source_code(f, 2, 4)
        assert result == "line2\nline3\nline4"

    def test_single_line(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("line1\nline2\nline3\n")
        result = extract_source_code(f, 1, 1)
        assert result == "line1"

    def test_nonexistent_file_returns_empty(self):
        result = extract_source_code(Path("/nonexistent/file.py"), 1, 5)
        assert result == ""

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.py"
        f.write_text("")
        result = extract_source_code(f, 1, 1)
        assert result == ""
