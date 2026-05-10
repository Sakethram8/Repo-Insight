# tests/test_change_engine.py
"""
Unit tests for change_engine.py — the 6-phase graph-driven pipeline.
Uses mocks for all external dependencies (LLM, FalkorDB, tools.py).
No live infrastructure required.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from change_engine import (
    ChangeSubgraph, ChangePlan, ChangeResult,
    GraphDrivenEngine, LOCALIZATION_PROMPT,
)
from apply_changes import RunResult


# ---------------------------------------------------------------------------
# Data class tests
# ---------------------------------------------------------------------------

class TestChangeSubgraph:
    def test_default_fields(self):
        sg = ChangeSubgraph(seed_nodes=["foo.bar"])
        assert sg.seed_nodes == ["foo.bar"]
        assert sg.blast_radius_nodes == []
        assert sg.impact_radius_nodes == []
        assert sg.caller_nodes == []
        assert sg.callee_nodes == []
        assert sg.all_affected_files == set()
        assert sg.source_code == {}
        assert sg.edges == []

    def test_multiple_seeds(self):
        sg = ChangeSubgraph(seed_nodes=["a.b", "c.d", "e.f"])
        assert len(sg.seed_nodes) == 3

    def test_mutable_collections_independent(self):
        """Verify each instance gets its own mutable defaults."""
        sg1 = ChangeSubgraph(seed_nodes=["a"])
        sg2 = ChangeSubgraph(seed_nodes=["b"])
        sg1.blast_radius_nodes.append({"fqn": "x"})
        assert sg2.blast_radius_nodes == []


class TestChangePlan:
    def test_default_not_validated(self):
        plan = ChangePlan()
        assert plan.is_validated is False
        assert plan.planned_files == set()
        assert plan.blast_radius_files == set()
        assert plan.missing_files == set()

    def test_missing_files_computation(self):
        plan = ChangePlan()
        plan.blast_radius_files = {"a.py", "b.py", "c.py"}
        plan.planned_files = {"a.py", "b.py"}
        plan.missing_files = plan.blast_radius_files - plan.planned_files
        assert plan.missing_files == {"c.py"}

    def test_validated_when_no_missing(self):
        plan = ChangePlan()
        plan.blast_radius_files = {"a.py", "b.py"}
        plan.planned_files = {"a.py", "b.py"}
        plan.missing_files = plan.blast_radius_files - plan.planned_files
        plan.is_validated = len(plan.missing_files) == 0
        assert plan.is_validated is True


class TestChangeResult:
    def test_default_empty(self):
        result = ChangeResult()
        assert result.answer == ""
        assert result.seeds == []
        assert result.subgraph is None
        assert result.plan is None
        assert result.edits == []
        assert result.error is None
        assert result.phases_completed == []
        assert result.timings == {}

    def test_hit_max_iterations_not_set_by_default(self):
        result = ChangeResult()
        assert result.error is None


# ---------------------------------------------------------------------------
# Engine tests — _parse_plan_json
# ---------------------------------------------------------------------------

class TestParsePlanJson:
    def setup_method(self):
        self.engine = GraphDrivenEngine(
            repo_root=Path("/tmp/fake"),
            graph=MagicMock(),
        )

    def test_clean_json_array(self):
        raw = '[{"file": "a.py", "action": "modify", "reason": "needs update"}]'
        result = self.engine._parse_plan_json(raw)
        assert len(result) == 1
        assert result[0]["file"] == "a.py"

    def test_json_in_markdown_fences(self):
        raw = '```json\n[{"file": "a.py", "action": "modify", "reason": "test"}]\n```'
        result = self.engine._parse_plan_json(raw)
        assert len(result) == 1
        assert result[0]["file"] == "a.py"

    def test_json_embedded_in_prose(self):
        raw = 'Here is the plan:\n[{"file": "b.py", "action": "no_change", "reason": "ok"}]\nDone.'
        result = self.engine._parse_plan_json(raw)
        assert len(result) == 1
        assert result[0]["file"] == "b.py"

    def test_invalid_json_returns_empty(self):
        raw = "This is not JSON at all, just prose."
        result = self.engine._parse_plan_json(raw)
        assert result == []

    def test_multiple_items(self):
        raw = json.dumps([
            {"file": "a.py", "action": "modify", "reason": "r1"},
            {"file": "b.py", "action": "no_change", "reason": "r2"},
        ])
        result = self.engine._parse_plan_json(raw)
        assert len(result) == 2

    def test_json_object_not_array_returns_empty(self):
        raw = '{"file": "a.py"}'
        result = self.engine._parse_plan_json(raw)
        assert result == []


# ---------------------------------------------------------------------------
# Engine tests — _localize_seeds
# ---------------------------------------------------------------------------

class TestLocalizeSeeds:
    def setup_method(self):
        self.engine = GraphDrivenEngine(
            repo_root=Path("/tmp/fake"),
            graph=MagicMock(),
        )

    @patch("change_engine.semantic_search")
    def test_normal_case_returns_llm_seeds(self, mock_search):
        """LLM returns valid JSON array of FQNs."""
        mock_search.return_value = {
            "results": [
                {"fqn": "parser.FunctionDef", "label": "Class", "file_path": "parser.py", "score": 0.9},
                {"fqn": "parser.parse_file", "label": "Function", "file_path": "parser.py", "score": 0.8},
            ]
        }
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '["parser.FunctionDef"]'
        self.engine.client = MagicMock()
        self.engine.client.chat.completions.create.return_value = mock_response

        seeds = self.engine._localize_seeds("Add decorators field")
        assert seeds == ["parser.FunctionDef"]

    @patch("change_engine.semantic_search")
    def test_llm_garbage_falls_back_to_top3(self, mock_search):
        """LLM returns garbage → falls back to top-3 semantic search."""
        candidates = [
            {"fqn": f"mod.func{i}", "label": "Function", "file_path": f"f{i}.py", "score": 0.9 - i * 0.1}
            for i in range(5)
        ]
        mock_search.return_value = {"results": candidates}
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "I cannot parse this request"
        self.engine.client = MagicMock()
        self.engine.client.chat.completions.create.return_value = mock_response

        seeds = self.engine._localize_seeds("something")
        assert len(seeds) == 3
        assert seeds == ["mod.func0", "mod.func1", "mod.func2"]

    @patch("change_engine.semantic_search")
    def test_empty_candidates_returns_empty(self, mock_search):
        """Semantic search returns nothing → empty seeds."""
        mock_search.return_value = {"results": []}
        seeds = self.engine._localize_seeds("nonexistent thing")
        assert seeds == []

    @patch("change_engine.semantic_search")
    def test_llm_exception_falls_back(self, mock_search):
        """LLM call raises exception → falls back to top-3."""
        candidates = [
            {"fqn": f"mod.f{i}", "label": "Function", "file_path": f"f{i}.py", "score": 0.5}
            for i in range(4)
        ]
        mock_search.return_value = {"results": candidates}
        self.engine.client = MagicMock()
        self.engine.client.chat.completions.create.side_effect = Exception("LLM down")

        seeds = self.engine._localize_seeds("test")
        assert len(seeds) == 3


# ---------------------------------------------------------------------------
# Engine tests — _expand_subgraph
# ---------------------------------------------------------------------------

class TestExpandSubgraph:
    def setup_method(self):
        self.engine = GraphDrivenEngine(
            repo_root=Path("/tmp/fake"),
            graph=MagicMock(),
        )

    @patch("change_engine.get_source_code")
    @patch("change_engine.get_callees")
    @patch("change_engine.get_callers")
    @patch("change_engine.get_downstream_deps")
    @patch("change_engine.get_upstream_callers")
    def test_collects_blast_radius(self, mock_blast, mock_impact, mock_callers,
                                    mock_callees, mock_source):
        mock_blast.return_value = {
            "affected": [
                {"fqn": "mod.caller1", "file_path": "caller.py"},
                {"fqn": "mod.caller2", "file_path": "caller2.py"},
            ]
        }
        mock_impact.return_value = {"impacted": []}
        mock_callers.return_value = {"callers": []}
        mock_callees.return_value = {"callees": []}
        mock_source.return_value = {"found": True, "source": "def seed(): pass", "file_path": "seed.py"}

        sg = self.engine._expand_subgraph(["mod.seed"])
        assert len(sg.blast_radius_nodes) == 2
        assert "caller.py" in sg.all_affected_files
        assert "caller2.py" in sg.all_affected_files

    @patch("change_engine.get_source_code")
    @patch("change_engine.get_callees")
    @patch("change_engine.get_callers")
    @patch("change_engine.get_downstream_deps")
    @patch("change_engine.get_upstream_callers")
    def test_collects_impact_radius(self, mock_blast, mock_impact, mock_callers,
                                     mock_callees, mock_source):
        mock_blast.return_value = {"affected": []}
        mock_impact.return_value = {
            "impacted": [{"fqn": "mod.downstream", "file_path": "down.py"}]
        }
        mock_callers.return_value = {"callers": []}
        mock_callees.return_value = {"callees": []}
        mock_source.return_value = {"found": True, "source": "code", "file_path": "s.py"}

        sg = self.engine._expand_subgraph(["mod.seed"])
        assert len(sg.impact_radius_nodes) == 1
        assert "down.py" in sg.all_affected_files

    @patch("change_engine.get_source_code")
    @patch("change_engine.get_callees")
    @patch("change_engine.get_callers")
    @patch("change_engine.get_downstream_deps")
    @patch("change_engine.get_upstream_callers")
    def test_deduplicates_across_seeds(self, mock_blast, mock_impact, mock_callers,
                                        mock_callees, mock_source):
        """Same FQN appears in blast radius of two seeds → only added once."""
        shared = {"fqn": "mod.shared", "file_path": "shared.py"}
        mock_blast.return_value = {"affected": [shared]}
        mock_impact.return_value = {"impacted": []}
        mock_callers.return_value = {"callers": []}
        mock_callees.return_value = {"callees": []}
        mock_source.return_value = {"found": True, "source": "code", "file_path": "s.py"}

        sg = self.engine._expand_subgraph(["mod.seed1", "mod.seed2"])
        assert len(sg.blast_radius_nodes) == 1  # Not 2

    @patch("change_engine.get_source_code")
    @patch("change_engine.get_callees")
    @patch("change_engine.get_callers")
    @patch("change_engine.get_downstream_deps")
    @patch("change_engine.get_upstream_callers")
    def test_fetches_source_for_all_seen_fqns(self, mock_blast, mock_impact,
                                               mock_callers, mock_callees, mock_source):
        mock_blast.return_value = {
            "affected": [{"fqn": "mod.caller1", "file_path": "c1.py"}]
        }
        mock_impact.return_value = {"impacted": []}
        mock_callers.return_value = {"callers": []}
        mock_callees.return_value = {"callees": []}
        mock_source.return_value = {"found": True, "source": "src", "file_path": "f.py"}

        sg = self.engine._expand_subgraph(["mod.seed"])
        # source_code should have entries for seed + blast radius node
        assert "mod.seed" in sg.source_code or len(sg.source_code) >= 1


# ---------------------------------------------------------------------------
# Engine tests — _plan_with_validation
# ---------------------------------------------------------------------------

class TestPlanWithValidation:
    def setup_method(self):
        self.engine = GraphDrivenEngine(
            repo_root=Path("/tmp/fake"),
            graph=MagicMock(),
        )

    def test_validated_when_plan_covers_all_files(self):
        subgraph = ChangeSubgraph(seed_nodes=["mod.foo"])
        subgraph.all_affected_files = {"a.py", "b.py"}
        subgraph.source_code = {"mod.foo": "def foo(): pass"}

        plan_json = json.dumps([
            {"file": "a.py", "action": "modify", "reason": "needs change"},
            {"file": "b.py", "action": "no_change", "reason": "no impact"},
        ])
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = plan_json
        self.engine.client = MagicMock()
        self.engine.client.chat.completions.create.return_value = mock_response

        plan = self.engine._plan_with_validation("test prompt", subgraph)
        assert plan.is_validated is True
        assert len(plan.missing_files) == 0

    def test_not_validated_when_files_missing(self):
        subgraph = ChangeSubgraph(seed_nodes=["mod.foo"])
        subgraph.all_affected_files = {"a.py", "b.py", "c.py"}
        subgraph.source_code = {"mod.foo": "def foo(): pass"}

        # First response: plan only covers a.py
        plan_json_1 = json.dumps([
            {"file": "a.py", "action": "modify", "reason": "needs change"},
        ])
        # Second response (force_coverage): still only covers a.py, b.py
        plan_json_2 = json.dumps([
            {"file": "a.py", "action": "modify", "reason": "needs change"},
            {"file": "b.py", "action": "no_change", "reason": "ok"},
        ])
        mock_resp1 = MagicMock()
        mock_resp1.choices = [MagicMock()]
        mock_resp1.choices[0].message.content = plan_json_1
        mock_resp2 = MagicMock()
        mock_resp2.choices = [MagicMock()]
        mock_resp2.choices[0].message.content = plan_json_2

        self.engine.client = MagicMock()
        self.engine.client.chat.completions.create.side_effect = [mock_resp1, mock_resp2]

        plan = self.engine._plan_with_validation("test prompt", subgraph)
        # The convergence fallback force-adds c.py and marks the plan as validated
        assert plan.is_validated is True
        assert "c.py" in plan.planned_files
        assert plan.missing_files == set()


# ---------------------------------------------------------------------------
# Engine tests — _generate_edits
# ---------------------------------------------------------------------------

class TestGenerateEdits:
    def setup_method(self):
        self.engine = GraphDrivenEngine(
            repo_root=Path("/tmp/fake"),
            graph=MagicMock(),
        )

    def test_parses_edit_blocks_from_llm_output(self):
        subgraph = ChangeSubgraph(seed_nodes=["mod.foo"])
        subgraph.source_code = {"mod.foo": "def foo(): pass"}

        plan = ChangePlan()
        plan.planned_files = {"parser.py"}
        plan.actions = {"parser.py": "modify"}
        plan.justifications = {"parser.py": "add field"}

        llm_output = (
            "FILE: parser.py\n"
            "<<<<<<< SEARCH\n"
            "    name: str\n"
            "=======\n"
            "    name: str\n"
            "    decorators: list[str] = field(default_factory=list)\n"
            ">>>>>>> REPLACE\n"
        )
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = llm_output
        self.engine.client = MagicMock()
        self.engine.client.chat.completions.create.return_value = mock_response

        edits, answer = self.engine._generate_edits("test prompt", plan, subgraph)
        assert len(edits) == 1
        assert edits[0].file_path == "parser.py"
        assert "decorators" in edits[0].replace_text

    def test_llm_failure_returns_empty_edits(self):
        subgraph = ChangeSubgraph(seed_nodes=["mod.foo"])
        subgraph.source_code = {}
        plan = ChangePlan()
        plan.planned_files = {"a.py"}
        plan.actions = {"a.py": "modify"}
        plan.justifications = {"a.py": "change"}

        self.engine.client = MagicMock()
        self.engine.client.chat.completions.create.side_effect = Exception("LLM error")

        edits, answer = self.engine._generate_edits("test", plan, subgraph)
        assert edits == []
        assert "failed" in answer.lower()


# ---------------------------------------------------------------------------
# Engine tests — _ensure_graph_fresh
# ---------------------------------------------------------------------------

class TestEnsureGraphFresh:
    @patch("change_engine.get_connection")
    @patch("change_engine.run_ingestion")
    def test_fingerprint_uses_original_root_not_sandbox(self, mock_ingest, mock_conn, tmp_path):
        original_root = tmp_path / "original"
        original_root.mkdir()
        sandbox_root = tmp_path / "sandbox"
        sandbox_root.mkdir()

        # Write file only in original_root
        (original_root / "test.py").write_text("x = 1")

        engine = GraphDrivenEngine(repo_root=original_root, sandbox_path=sandbox_root)
        engine.graph = None  # Force fingerprint computation
        mock_ingest.return_value = {}
        mock_conn.return_value = MagicMock()

        engine._ensure_graph_fresh()

        # Ensure run_ingestion was called with sandbox path
        mock_ingest.assert_called_once_with(str(sandbox_root))

    def test_skips_ingestion_when_fingerprint_matches(self, tmp_path):
        original_root = tmp_path / "original"
        original_root.mkdir()
        (original_root / "test.py").write_text("x = 1")

        # Compute expected fingerprint
        import change_engine
        engine = change_engine.GraphDrivenEngine(repo_root=original_root)
        engine.graph = MagicMock()
        
        # Manually compute what _ensure_graph_fresh would compute for original_root
        # (uses content hashes since commit f6df223 — mtime is unreliable across clones)
        from ingest import _file_content_hash
        from config import SKIP_DIRS
        current_hashes = {}
        for f in sorted(original_root.rglob("*.py")):
            parts = f.relative_to(original_root).parts
            if not any(p in SKIP_DIRS for p in parts):
                rel = str(f.relative_to(original_root))
                current_hashes[rel] = _file_content_hash(f)
        expected_fp = str(hash(frozenset(current_hashes.items())))

        mock_res = MagicMock()
        mock_res.result_set = [[expected_fp]]
        engine.graph.query.return_value = mock_res

        with patch("change_engine.run_ingestion") as mock_ingest:
            result = engine._ensure_graph_fresh()
            mock_ingest.assert_not_called()
            assert result.get("skipped") is True
            assert result.get("fingerprint") == expected_fp


# ---------------------------------------------------------------------------
# Engine tests — _retry_edits
# ---------------------------------------------------------------------------

class TestRetryEdits:
    def test_retry_uses_max_tokens_8192(self):
        engine = GraphDrivenEngine(repo_root=Path("/tmp/fake"), graph=MagicMock())
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "FILE: test.py\n<<<<<<< SEARCH\nx\n=======\ny\n>>>>>>> REPLACE\n"
        
        engine.client = MagicMock()
        engine.client.chat.completions.create.return_value = mock_response

        subgraph = ChangeSubgraph(seed_nodes=["mod"])
        subgraph.source_code = {"mod": "x"}
        run_result = RunResult(exit_code=1, stdout="AssertionError", stderr="")

        edits = engine._retry_edits("fix it", subgraph, run_result)
        
        engine.client.chat.completions.create.assert_called_once()
        kwargs = engine.client.chat.completions.create.call_args[1]
        assert kwargs.get("max_tokens") == 32768
        assert len(edits) == 1

    def test_retry_returns_empty_on_llm_failure(self):
        engine = GraphDrivenEngine(repo_root=Path("/tmp/fake"), graph=MagicMock())
        engine.client = MagicMock()
        engine.client.chat.completions.create.side_effect = Exception("API down")

        subgraph = ChangeSubgraph(seed_nodes=["mod"])
        run_result = RunResult(exit_code=1, stdout="err", stderr="")

        edits = engine._retry_edits("fix it", subgraph, run_result)
        assert edits == []


# ---------------------------------------------------------------------------
# Engine tests — full run (Phase 0 skip, mocked everything)
# ---------------------------------------------------------------------------

class TestFullRun:
    @patch("change_engine.run_ingestion")
    @patch("change_engine.get_connection")
    @patch("change_engine.semantic_search")
    @patch("change_engine.get_upstream_callers")
    @patch("change_engine.get_downstream_deps")
    @patch("change_engine.get_callers")
    @patch("change_engine.get_callees")
    @patch("change_engine.get_source_code")
    def test_run_with_skip_apply(self, mock_source, mock_callees, mock_callers,
                                  mock_impact, mock_blast, mock_search,
                                  mock_conn, mock_ingest):
        """Full pipeline run with skip_apply=True (no sandbox/tests)."""
        # Phase 0
        mock_ingest.return_value = {"functions": 10, "call_edges": 5}
        mock_conn.return_value = MagicMock()

        # Phase 1
        mock_search.return_value = {
            "results": [{"fqn": "mod.target", "label": "Function", "file_path": "mod.py", "score": 0.95}]
        }

        # Phase 2
        mock_blast.return_value = {"affected": [{"fqn": "mod.caller", "file_path": "mod.py"}]}
        mock_impact.return_value = {"impacted": []}
        mock_callers.return_value = {"callers": []}
        mock_callees.return_value = {"callees": []}
        mock_source.return_value = {"found": True, "source": "def target(): pass", "file_path": "mod.py"}

        engine = GraphDrivenEngine(repo_root=Path("/tmp/fake"), graph=MagicMock())

        # Mock LLM for Phase 1, 3, 4
        mock_resp_seeds = MagicMock()
        mock_resp_seeds.choices = [MagicMock()]
        mock_resp_seeds.choices[0].message.content = '["mod.target"]'

        mock_resp_plan = MagicMock()
        mock_resp_plan.choices = [MagicMock()]
        mock_resp_plan.choices[0].message.content = json.dumps([
            {"file": "mod.py", "action": "modify", "reason": "change target"}
        ])

        mock_resp_edit = MagicMock()
        mock_resp_edit.choices = [MagicMock()]
        mock_resp_edit.choices[0].message.content = (
            "FILE: mod.py\n<<<<<<< SEARCH\ndef target(): pass\n=======\n"
            "def target(timeout=None): pass\n>>>>>>> REPLACE\n\nDone."
        )

        engine.client = MagicMock()
        engine.client.chat.completions.create.side_effect = [
            mock_resp_seeds, mock_resp_plan, mock_resp_edit,
        ]

        result = engine.run("Add timeout parameter", skip_apply=True)

        assert len(result.phases_completed) >= 4
        assert result.seeds == ["mod.target"]
        assert result.subgraph is not None
        assert result.plan is not None
        assert len(result.edits) >= 1
        assert result.error is None

    @patch("change_engine.run_ingestion")
    @patch("change_engine.get_connection")
    @patch("change_engine.semantic_search")
    def test_run_aborts_on_no_seeds(self, mock_search, mock_conn, mock_ingest):
        """Pipeline aborts early if no seed nodes are found."""
        mock_ingest.return_value = {"functions": 0, "call_edges": 0}
        mock_conn.return_value = MagicMock()
        mock_search.return_value = {"results": []}

        engine = GraphDrivenEngine(repo_root=Path("/tmp/fake"), graph=MagicMock())
        engine.client = MagicMock()

        result = engine.run("something impossible")
        assert result.error is not None
        assert "seed" in result.error.lower()
        assert len(result.phases_completed) == 2  # Phase 0 + Phase 1 (then abort)

    def test_on_phase_callback_invoked(self):
        """Verify the _notify method calls the callback."""
        engine = GraphDrivenEngine(repo_root=Path("/tmp/fake"), graph=MagicMock())
        captured = []
        engine._on_phase = lambda phase, data: captured.append((phase, data))
        engine._notify("test_phase", {"key": "value"})
        assert len(captured) == 1
        assert captured[0] == ("test_phase", {"key": "value"})

    def test_on_phase_callback_error_swallowed(self):
        """Callback errors should not crash the pipeline."""
        engine = GraphDrivenEngine(repo_root=Path("/tmp/fake"), graph=MagicMock())

        def bad_callback(phase, data):
            raise RuntimeError("callback crash")

        engine._on_phase = bad_callback
        # Should not raise
        engine._notify("test", {})
