# tests/test_tools.py
"""
Integration tests for tools.py.
Require a live FalkorDB. Skip if FalkorDB is not reachable.
"""

import pytest
import falkordb
from pathlib import Path
from config import FALKORDB_HOST, FALKORDB_PORT, IMPACT_RADIUS_WARN_THRESHOLD, BLAST_RADIUS_MAX_DEPTH, IMPACT_RADIUS_MAX_DEPTH
from ingest import run_ingestion, get_connection, create_indices, ingest_parsed_files
from parser import parse_directory
from tools import (get_function_context, get_callers, get_callees,
                   get_source_code, semantic_search, get_downstream_deps, 
                   get_upstream_callers)
from unittest.mock import MagicMock

FIXTURES = str(Path(__file__).parent / "fixtures")

# Use a separate test graph name to avoid clobbering real data
TEST_GRAPH_NAME = "repo_insight_test"


@pytest.fixture(scope="module")
def graph():
    try:
        db = falkordb.FalkorDB(host=FALKORDB_HOST, port=FALKORDB_PORT)
        g = db.select_graph(TEST_GRAPH_NAME)
        # Flush and ingest test fixtures
        try:
            g.query("MATCH (n) DETACH DELETE n")
        except Exception:
            pass
        create_indices(g)
        # Store repo root so get_source_code can resolve relative paths
        fixtures_path = str(Path(FIXTURES).resolve())
        g.query(
            "MERGE (m:Meta {key: 'repo_root'}) SET m.value = $root",
            {"root": fixtures_path},
        )
        parsed = parse_directory(Path(FIXTURES))
        ingest_parsed_files(parsed, g, Path(FIXTURES))
        yield g
        # Teardown: drop test graph
        try:
            g.query("MATCH (n) DETACH DELETE n")
        except Exception:
            pass
    except Exception:
        pytest.skip("FalkorDB not reachable")


@pytest.mark.integration
class TestGetFunctionContext:
    def test_known_function_returns_found_true(self, graph):
        result = get_function_context("add", graph)
        assert result["found"] is True

    def test_unknown_function_returns_found_false(self, graph):
        result = get_function_context("nonexistent_xyz", graph)
        assert result["found"] is False

    def test_returns_correct_file_path(self, graph):
        result = get_function_context("add", graph)
        assert result["file_path"].endswith("simple_module.py")

    def test_returns_docstring(self, graph):
        result = get_function_context("add", graph)
        # docstring or summary should exist
        assert result.get("summary") is not None or result.get("docstring") is not None

    def test_returns_method_info(self, graph):
        result = get_function_context("add", graph)
        assert result["is_method"] is True
        assert result["class_name"] == "Calculator"


@pytest.mark.integration
class TestGetCallers:
    def test_wrapper_calls_standalone(self, graph):
        result = get_callers("standalone_function", graph)
        caller_names = [c["fqn"] for c in result["callers"]]
        assert any("wrapper" in name for name in caller_names)

    def test_function_with_no_callers_returns_empty_list(self, graph):
        result = get_callers("double_wrap", graph)
        assert result["caller_count"] == 0
        assert result["callers"] == []

    def test_caller_count_matches_list_length(self, graph):
        result = get_callers("standalone_function", graph)
        assert result["caller_count"] == len(result["callers"])


@pytest.mark.integration
class TestGetCallees:
    def test_alpha_calls_beta(self, graph):
        result = get_callees("alpha", graph)
        callee_names = [c["fqn"] for c in result["callees"]]
        assert any("beta" in name for name in callee_names)

    def test_gamma_has_no_callees(self, graph):
        result = get_callees("gamma", graph)
        assert result["callee_count"] == 0


@pytest.mark.integration
class TestGetDownstreamDeps:
    def test_alpha_impacts_beta_and_gamma(self, graph):
        result = get_downstream_deps("alpha", graph)
        fqns = [r["fqn"] for r in result["impacted"]]
        assert any("beta" in f for f in fqns)
        assert any("gamma" in f for f in fqns)

    def test_gamma_has_no_impact(self, graph):
        result = get_downstream_deps("gamma", graph)
        assert result["impacted_count"] == 0

    def test_result_limited_to_50(self, graph):
        result = get_downstream_deps("alpha", graph)
        assert len(result["impacted"]) <= 50

    def test_direction_is_downstream(self, graph):
        result = get_downstream_deps("alpha", graph)
        assert result["direction"] == "downstream"

    def test_warning_flag_when_threshold_exceeded(self, graph):
        result = get_downstream_deps("alpha", graph)
        if result["impacted_count"] > IMPACT_RADIUS_WARN_THRESHOLD:
            assert result["warning"] is True
        else:
            assert result["warning"] is False

    def test_depth_field_matches_parameter(self, graph):
        result = get_downstream_deps("alpha", graph, max_depth=3)
        assert result["depth"] == 3


@pytest.mark.integration
class TestGetUpstreamCallers:
    def test_gamma_blast_radius_includes_alpha_and_beta(self, graph):
        """gamma is called by beta, which is called by alpha."""
        result = get_upstream_callers("gamma", graph)
        fqns = [r["fqn"] for r in result["affected"]]
        assert any("beta" in f for f in fqns)
        # alpha -> beta -> gamma, so alpha should also appear at depth 2
        assert any("alpha" in f for f in fqns)

    def test_direction_is_upstream(self, graph):
        result = get_upstream_callers("gamma", graph)
        assert result["direction"] == "upstream"

    def test_alpha_has_no_upstream_callers(self, graph):
        """alpha is top of the chain, nothing calls it."""
        result = get_upstream_callers("alpha", graph)
        assert result["affected_count"] == 0

    def test_affected_count_matches_list_length(self, graph):
        result = get_upstream_callers("gamma", graph)
        assert result["affected_count"] == len(result["affected"])

    def test_result_limited_to_50(self, graph):
        result = get_upstream_callers("gamma", graph)
        assert len(result["affected"]) <= 50


@pytest.mark.integration
class TestGetSourceCode:
    def test_known_function_returns_source(self, graph):
        result = get_source_code("standalone_function", graph)
        assert result["found"] is True
        # The source should contain the actual function definition
        assert "def standalone_function" in result["source"]

    def test_unknown_function_returns_not_found(self, graph):
        result = get_source_code("nonexistent_xyz", graph)
        assert result["found"] is False

    def test_returns_correct_line_numbers(self, graph):
        result = get_source_code("standalone_function", graph)
        assert result["found"] is True
        assert result["start_line"] > 0
        assert result["end_line"] >= result["start_line"]

    def test_method_source_includes_self(self, graph):
        result = get_source_code("add", graph)
        if result["found"] and result["source"] != "<file not readable>":
            assert "self" in result["source"]


@pytest.mark.integration
class TestSemanticSearch:
    def test_returns_results_for_general_query(self, graph):
        result = semantic_search("arithmetic operations", graph)
        assert len(result["results"]) > 0

    def test_results_have_required_fields(self, graph):
        result = semantic_search("function that squares a number", graph)
        for r in result["results"]:
            assert "label" in r
            assert "fqn" in r
            assert "file_path" in r
            assert "score" in r
            assert r["label"] in ("Function", "Class")

    def test_top_result_relevance(self, graph):
        result = semantic_search("square a number", graph)
        if result["results"]:
            top_fqns = [r["fqn"] for r in result["results"][:3]]
            assert any("standalone_function" in f for f in top_fqns)

    def test_respects_top_k(self, graph):
        result = semantic_search("test", graph, top_k=2)
        assert len(result["results"]) <= 2


# ---------------------------------------------------------------------------
# Default Depth Parameter Tests
# ---------------------------------------------------------------------------

class TestDepthDefaults:
    def test_get_upstream_callers_default_depth_is_blast_radius(self):
        mock_graph = MagicMock()
        mock_graph.query.return_value = MagicMock(result_set=[])
        
        get_upstream_callers("some.func", mock_graph)
        
        # Check what was passed to graph.query
        call_args = mock_graph.query.call_args[0]
        cypher_query = call_args[0]
        
        assert f"CALLS*1..{BLAST_RADIUS_MAX_DEPTH}" in cypher_query
        if BLAST_RADIUS_MAX_DEPTH != IMPACT_RADIUS_MAX_DEPTH:
            assert f"CALLS*1..{IMPACT_RADIUS_MAX_DEPTH}" not in cypher_query

    def test_get_downstream_deps_default_depth_is_impact_radius(self):
        mock_graph = MagicMock()
        mock_graph.query.return_value = MagicMock(result_set=[])
        
        get_downstream_deps("some.func", mock_graph)
        
        # Check what was passed to graph.query
        call_args = mock_graph.query.call_args[0]
        cypher_query = call_args[0]
        
        assert f"CALLS*1..{IMPACT_RADIUS_MAX_DEPTH}" in cypher_query
        if BLAST_RADIUS_MAX_DEPTH != IMPACT_RADIUS_MAX_DEPTH:
            assert f"CALLS*1..{BLAST_RADIUS_MAX_DEPTH}" not in cypher_query

    def test_depths_are_different_values(self):
        """Canary test: if they are the same, the above directional assertions lose meaning."""
        assert BLAST_RADIUS_MAX_DEPTH != IMPACT_RADIUS_MAX_DEPTH

