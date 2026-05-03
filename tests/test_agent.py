# tests/test_agent.py
"""
Unit tests for agent.py.
Uses pytest-mock to mock tools.py functions. Does NOT call the real LLM or FalkorDB.
"""

import json
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from agent import run_repo_agent, _dispatch_tool, AGENT_MAX_ITERATIONS


class TestDispatchTool:
    def test_unknown_tool_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown tool"):
            _dispatch_tool("nonexistent_tool", {}, MagicMock())

    def test_dispatches_get_function_context(self, mocker):
        mock_graph = MagicMock()
        mock_result = {"found": True, "name": "foo"}
        mocker.patch("agent.get_function_context", return_value=mock_result)
        result = _dispatch_tool("get_function_context", {"name": "foo"}, mock_graph)
        parsed = json.loads(result)
        assert parsed == mock_result

    def test_dispatches_get_callers(self, mocker):
        mock_graph = MagicMock()
        mock_result = {"target": "foo", "caller_count": 1, "callers": [{"name": "bar"}]}
        mocker.patch("agent.get_callers", return_value=mock_result)
        result = _dispatch_tool("get_callers", {"name": "foo"}, mock_graph)
        parsed = json.loads(result)
        assert parsed["target"] == "foo"

    def test_dispatches_get_callees(self, mocker):
        mock_graph = MagicMock()
        mock_result = {"target": "foo", "callee_count": 0, "callees": []}
        mocker.patch("agent.get_callees", return_value=mock_result)
        result = _dispatch_tool("get_callees", {"name": "foo"}, mock_graph)
        parsed = json.loads(result)
        assert parsed["callee_count"] == 0

    def test_dispatches_get_downstream_deps(self, mocker):
        mock_graph = MagicMock()
        mock_result = {"source": "foo", "depth": 2, "impacted_count": 0,
                       "warning": False, "impacted": []}
        mocker.patch("agent.get_downstream_deps", return_value=mock_result)
        result = _dispatch_tool("get_downstream_deps", {"name": "foo"}, mock_graph)
        parsed = json.loads(result)
        assert parsed["source"] == "foo"

    def test_dispatches_semantic_search(self, mocker):
        mock_graph = MagicMock()
        mock_result = {"query": "test", "results": []}
        mocker.patch("agent.semantic_search", return_value=mock_result)
        result = _dispatch_tool("semantic_search", {"query": "test"}, mock_graph)
        parsed = json.loads(result)
        assert parsed["query"] == "test"

    def test_dispatches_get_upstream_callers(self, mocker):
        mock_graph = MagicMock()
        mock_result = {"target": "foo", "direction": "upstream", "depth": 2,
                       "affected_count": 3, "warning": False, "affected": []}
        mocker.patch("agent.get_upstream_callers", return_value=mock_result)
        result = _dispatch_tool("get_upstream_callers", {"name": "foo"}, mock_graph)
        parsed = json.loads(result)
        assert parsed["target"] == "foo"
        assert parsed["direction"] == "upstream"

    def test_dispatches_get_source_code(self, mocker):
        mock_graph = MagicMock()
        mock_result = {"found": True, "name": "foo", "source": "def foo(): pass"}
        mocker.patch("agent.get_source_code", return_value=mock_result)
        result = _dispatch_tool("get_source_code", {"name": "foo"}, mock_graph, iteration_architect_calls=2)
        parsed = json.loads(result)
        assert parsed["found"] is True
        assert "def foo" in parsed["source"]


def _make_mock_response(content=None, tool_calls=None):
    """Helper to create a mock OpenAI ChatCompletion response."""
    mock_message = MagicMock()
    mock_message.content = content
    mock_message.tool_calls = tool_calls

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    return mock_response


def _make_mock_tool_call(tc_id, name, arguments):
    """Helper to create a mock tool_call object."""
    mock_tc = MagicMock()
    mock_tc.id = tc_id
    mock_tc.function.name = name
    mock_tc.function.arguments = json.dumps(arguments)
    return mock_tc


class TestRunRepoAgent:
    def test_returns_answer_when_no_tool_calls(self, mocker):
        """LLM gives a direct answer without any tool calls."""
        mock_graph = MagicMock()
        mock_response = _make_mock_response(
            content="This is a direct answer.", tool_calls=None
        )

        mock_client_instance = MagicMock()
        mock_client_instance.chat.completions.create.return_value = mock_response
        mocker.patch("agent.openai.OpenAI", return_value=mock_client_instance)

        result = run_repo_agent("What is foo?", mock_graph)
        assert result["answer"] == "This is a direct answer."
        assert result["tool_calls_log"] == []
        assert result["hit_max_iterations"] is False

    def test_tool_call_dispatched_and_appended(self, mocker):
        """LLM makes one tool call, then gives a final answer."""
        mock_graph = MagicMock()

        # First response: tool call
        tc = _make_mock_tool_call("tc_1", "get_function_context", {"name": "foo"})
        resp1 = _make_mock_response(content="", tool_calls=[tc])

        # Second response: final answer
        resp2 = _make_mock_response(content="Here is the plan.", tool_calls=None)

        mock_client_instance = MagicMock()
        mock_client_instance.chat.completions.create.side_effect = [resp1, resp2]
        mocker.patch("agent.openai.OpenAI", return_value=mock_client_instance)

        # Mock the tool function
        mocker.patch(
            "agent.get_function_context",
            return_value={"found": True, "name": "foo"},
        )

        result = run_repo_agent("Tell me about foo", mock_graph)
        assert len(result["tool_calls_log"]) == 1
        assert result["tool_calls_log"][0]["tool"] == "get_function_context"
        assert result["answer"] == "Here is the plan."
        assert result["iterations"] == 2

    def test_max_iterations_guard(self, mocker):
        """LLM always returns tool calls, never a final answer."""
        mock_graph = MagicMock()

        # Create a tool call response that repeats forever
        tc = _make_mock_tool_call("tc_loop", "get_function_context", {"name": "x"})
        loop_response = _make_mock_response(content="", tool_calls=[tc])

        mock_client_instance = MagicMock()
        mock_client_instance.chat.completions.create.return_value = loop_response
        mocker.patch("agent.openai.OpenAI", return_value=mock_client_instance)
        mocker.patch(
            "agent.get_function_context",
            return_value={"found": False, "name": "x"},
        )

        result = run_repo_agent("Infinite loop test", mock_graph)
        assert result["hit_max_iterations"] is True
        assert result["iterations"] == AGENT_MAX_ITERATIONS

    def test_hit_max_iterations_returns_error_message(self, mocker):
        """Verify the answer field contains a human-readable error, not a traceback."""
        mock_graph = MagicMock()

        tc = _make_mock_tool_call("tc_loop", "get_callers", {"name": "y"})
        loop_response = _make_mock_response(content="", tool_calls=[tc])

        mock_client_instance = MagicMock()
        mock_client_instance.chat.completions.create.return_value = loop_response
        mocker.patch("agent.openai.OpenAI", return_value=mock_client_instance)
        mocker.patch(
            "agent.get_callers",
            return_value={"target": "y", "caller_count": 0, "callers": []},
        )

        result = run_repo_agent("Loop error test", mock_graph)
        assert result["hit_max_iterations"] is True
        assert "maximum iteration limit" in result["answer"].lower()
        assert "Traceback" not in result["answer"]

    def test_multiple_tool_calls_in_single_iteration(self, mocker):
        """LLM makes two tool calls in one message, then gives a final answer."""
        mock_graph = MagicMock()

        tc1 = _make_mock_tool_call("tc_a", "get_function_context", {"name": "a"})
        tc2 = _make_mock_tool_call("tc_b", "get_callers", {"name": "b"})
        resp1 = _make_mock_response(content="", tool_calls=[tc1, tc2])
        resp2 = _make_mock_response(content="Final.", tool_calls=None)

        mock_client_instance = MagicMock()
        mock_client_instance.chat.completions.create.side_effect = [resp1, resp2]
        mocker.patch("agent.openai.OpenAI", return_value=mock_client_instance)
        mocker.patch(
            "agent.get_function_context",
            return_value={"found": True, "name": "a"},
        )
        mocker.patch(
            "agent.get_callers",
            return_value={"target": "b", "caller_count": 0, "callers": []},
        )

        result = run_repo_agent("Multi tool", mock_graph)
        assert len(result["tool_calls_log"]) == 2
        assert result["answer"] == "Final."
