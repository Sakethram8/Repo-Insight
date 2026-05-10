# tests/test_ingest.py
"""
Unit tests for ingest.py.
Uses mocks for FalkorDB and LLM calls. Tests helpers and ingestion logic.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from ingest import (
    _file_to_module, generate_summaries_batch, extract_source_code,
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


class TestGenerateSummariesBatch:
    def test_empty_batch_returns_empty_dict(self):
        result = generate_summaries_batch([])
        assert result == {}

    @patch("ingest._get_summary_client")
    def test_successful_summary(self, mock_client_factory):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"123": "This function adds two numbers."}'
        mock_client.chat.completions.create.return_value = mock_response
        mock_client_factory.return_value = mock_client

        result = generate_summaries_batch([("123", "def add(a, b): return a + b", "", False)])
        assert result == {"123": "This function adds two numbers."}

    @patch("ingest._get_summary_client")
    def test_llm_error_returns_empty_dict(self, mock_client_factory):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("LLM down")
        mock_client_factory.return_value = mock_client

        result = generate_summaries_batch([("123", "def broken(): pass")])
        assert result == {}


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
