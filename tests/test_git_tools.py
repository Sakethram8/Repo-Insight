# tests/test_git_tools.py
"""Unit tests for git_tools.py — no FalkorDB required."""

from unittest.mock import MagicMock, patch

import pytest

from git_tools import (
    _diff_signatures,
    _get_changed_files,
    _resolve_repo_root,
    git_diff_impact,
)


# ---------------------------------------------------------------------------
# _diff_signatures
# ---------------------------------------------------------------------------

class TestDiffSignatures:
    def _iface(self, fns):
        return {"functions": fns}

    def _fn(self, fqn, params=None, return_annotation=None):
        return {"fqn": fqn, "params": params or [], "return_annotation": return_annotation}

    def test_no_changes(self):
        old = self._iface([self._fn("mod.foo", ["x", "y"])])
        new = self._iface([self._fn("mod.foo", ["x", "y"])])
        changed, removed = _diff_signatures(old, new)
        assert changed == []
        assert removed == []

    def test_params_changed(self):
        old = self._iface([self._fn("mod.foo", ["x"])])
        new = self._iface([self._fn("mod.foo", ["x", "y"])])
        changed, removed = _diff_signatures(old, new)
        assert len(changed) == 1
        assert changed[0]["fqn"] == "mod.foo"
        assert changed[0]["old_params"] == ["x"]
        assert changed[0]["new_params"] == ["x", "y"]

    def test_return_changed(self):
        old = self._iface([self._fn("mod.foo", ["x"], "str")])
        new = self._iface([self._fn("mod.foo", ["x"], "int")])
        changed, removed = _diff_signatures(old, new)
        assert len(changed) == 1
        assert changed[0]["old_return"] == "str"
        assert changed[0]["new_return"] == "int"

    def test_function_removed(self):
        old = self._iface([self._fn("mod.foo"), self._fn("mod.bar")])
        new = self._iface([self._fn("mod.foo")])
        changed, removed = _diff_signatures(old, new)
        assert changed == []
        assert "mod.bar" in removed

    def test_function_added_not_reported(self):
        old = self._iface([self._fn("mod.foo")])
        new = self._iface([self._fn("mod.foo"), self._fn("mod.bar")])
        changed, removed = _diff_signatures(old, new)
        assert changed == []
        assert removed == []

    def test_empty_old_iface(self):
        old = self._iface([])
        new = self._iface([self._fn("mod.foo")])
        changed, removed = _diff_signatures(old, new)
        assert changed == []
        assert removed == []

    def test_none_return_treated_as_empty(self):
        old = self._iface([self._fn("mod.foo", [], None)])
        new = self._iface([self._fn("mod.foo", [], "")])
        changed, removed = _diff_signatures(old, new)
        # None and "" are both normalised to "" by tools.py; no change
        assert changed == []


# ---------------------------------------------------------------------------
# _get_changed_files
# ---------------------------------------------------------------------------

class TestGetChangedFiles:
    def test_splits_modified_and_deleted(self, tmp_path):
        def fake_check_output(cmd, **kwargs):
            if "--diff-filter=ACMR" in cmd:
                return "api.py\nconfig.py\nREADME.md\n"
            if "--diff-filter=D" in cmd:
                return "old_module.py\n"
            return ""

        with patch("git_tools.subprocess.check_output", side_effect=fake_check_output):
            modified, deleted = _get_changed_files("HEAD", tmp_path)

        assert "api.py" in modified
        assert "config.py" in modified
        assert "README.md" not in modified   # not a .py file
        assert "old_module.py" in deleted

    def test_empty_diff(self, tmp_path):
        with patch("git_tools.subprocess.check_output", return_value=""):
            modified, deleted = _get_changed_files("HEAD", tmp_path)
        assert modified == []
        assert deleted == []


# ---------------------------------------------------------------------------
# _resolve_repo_root
# ---------------------------------------------------------------------------

class TestResolveRepoRoot:
    def test_explicit_arg_wins(self, tmp_path):
        graph = MagicMock()
        result = _resolve_repo_root(str(tmp_path), graph)
        assert result == tmp_path

    def test_falls_back_to_git(self, tmp_path):
        graph = MagicMock()
        with patch("git_tools.subprocess.check_output", return_value=str(tmp_path) + "\n"):
            result = _resolve_repo_root(None, graph)
        assert result == tmp_path

    def test_falls_back_to_graph_meta(self, tmp_path):
        import subprocess as _sp
        graph = MagicMock()
        graph.query.return_value.result_set = [[str(tmp_path)]]
        with patch("git_tools.subprocess.check_output",
                   side_effect=_sp.CalledProcessError(128, "git")):
            result = _resolve_repo_root(None, graph)
        assert result == tmp_path

    def test_raises_when_all_fail(self):
        import subprocess as _sp
        graph = MagicMock()
        graph.query.return_value.result_set = []
        with patch("git_tools.subprocess.check_output",
                   side_effect=_sp.CalledProcessError(128, "git")):
            with pytest.raises(ValueError, match="repo_root"):
                _resolve_repo_root(None, graph)


# ---------------------------------------------------------------------------
# git_diff_impact (integration-style with mocked graph + subprocess)
# ---------------------------------------------------------------------------

class TestGitDiffImpact:
    def _make_graph(self):
        g = MagicMock()
        g.query.return_value.result_set = []   # Meta query returns nothing
        return g

    def _iface(self, fns):
        return {"functions": fns, "function_count": len(fns)}

    def test_no_changed_files_returns_empty(self, tmp_path):
        graph = self._make_graph()
        with (
            patch("git_tools.subprocess.check_output", return_value=""),
            patch("git_tools._resolve_repo_root", return_value=tmp_path),
        ):
            result = git_diff_impact("HEAD", repo_root=tmp_path, graph=graph)

        assert result["changed_files"] == []
        assert result["deleted_files"] == []
        assert result["total_at_risk"] == 0

    def test_interface_change_reported(self, tmp_path):
        graph = self._make_graph()

        old_iface = self._iface([{"fqn": "mod.foo", "params": ["x"],
                                   "return_annotation": None, "name": "foo"}])
        new_iface = self._iface([{"fqn": "mod.foo", "params": ["x", "y"],
                                   "return_annotation": None, "name": "foo"}])
        impact_result = {
            "file_path": "mod.py",
            "interface_breaking_changes": 1,
            "impact": [{
                "fqn": "mod.foo",
                "change": {"params_changed": True, "return_changed": False,
                           "old_params": ["x"], "new_params": ["x", "y"],
                           "old_return": None, "new_return": None},
                "at_risk_callers": [{"fqn": "other.bar", "file_path": "other.py",
                                     "start_line": 5, "module_name": "other"}],
                "caller_count": 1,
            }],
        }

        def fake_check_output(cmd, **kwargs):
            if "--diff-filter=ACMR" in cmd:
                return "mod.py\n"
            return ""

        with (
            patch("git_tools.subprocess.check_output", side_effect=fake_check_output),
            patch("git_tools._resolve_repo_root", return_value=tmp_path),
            patch("git_tools.get_file_interface", side_effect=[old_iface, new_iface]),
            patch("git_tools.analyze_edit_impact", return_value=impact_result),
            patch("git_tools.reingest_files"),
        ):
            result = git_diff_impact("HEAD", repo_root=tmp_path, graph=graph)

        assert result["total_at_risk"] == 1
        assert len(result["interface_breaks"]) == 1
        assert result["interface_breaks"][0]["fqn"] == "mod.foo"

    def test_deleted_file_breaks_reported(self, tmp_path):
        graph = self._make_graph()

        deleted_iface = self._iface([{"fqn": "gone.helper", "params": [],
                                      "return_annotation": None, "name": "helper"}])
        callers_result = {
            "target_fqn": "gone.helper",
            "cross_module_caller_count": 2,
            "callers": [
                {"fqn": "api.view", "file_path": "api.py",
                 "start_line": 10, "module_name": "api"},
                {"fqn": "cli.run", "file_path": "cli.py",
                 "start_line": 20, "module_name": "cli"},
            ],
        }

        def fake_check_output(cmd, **kwargs):
            if "--diff-filter=D" in cmd:
                return "gone.py\n"
            return ""

        with (
            patch("git_tools.subprocess.check_output", side_effect=fake_check_output),
            patch("git_tools._resolve_repo_root", return_value=tmp_path),
            patch("git_tools.get_file_interface", return_value=deleted_iface),
            patch("git_tools.get_cross_module_callers", return_value=callers_result),
            patch("git_tools.reingest_files"),
        ):
            result = git_diff_impact("HEAD", repo_root=tmp_path, graph=graph)

        assert result["total_at_risk"] == 2
        assert len(result["deleted_function_breaks"]) == 1
        assert result["deleted_function_breaks"][0]["deleted_fqn"] == "gone.helper"
        assert result["deleted_function_breaks"][0]["caller_count"] == 2

    def test_git_error_returns_error_key(self, tmp_path):
        import subprocess
        graph = self._make_graph()
        with (
            patch("git_tools.subprocess.check_output",
                  side_effect=subprocess.CalledProcessError(128, "git")),
            patch("git_tools._resolve_repo_root", return_value=tmp_path),
        ):
            result = git_diff_impact("HEAD", repo_root=tmp_path, graph=graph)

        assert "error" in result
        assert result["total_at_risk"] == 0
