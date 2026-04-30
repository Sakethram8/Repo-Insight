# tests/test_scoring.py
"""
Unit tests for scoring.py — file extraction, precision/recall/F1,
ground truth validation, and hallucination detection.
No live infrastructure required.
"""

import pytest
from scoring import (
    extract_mentioned_files, score_single,
    GROUND_TRUTH_TASKS, TaskScore, ScoringReport,
)


# ---------------------------------------------------------------------------
# File extraction from LLM output
# ---------------------------------------------------------------------------

class TestExtractMentionedFiles:
    def setup_method(self):
        self.known_files = {
            "parser.py", "ingest.py", "tools.py", "agent.py",
            "config.py", "tests/test_parser.py", "mcp_server.py",
            "change_engine.py", "graph_health.py", "app.py",
        }

    def test_extracts_basename_mentions(self):
        answer = "You need to modify parser.py and also update ingest.py."
        result = extract_mentioned_files(answer, self.known_files)
        assert "parser.py" in result
        assert "ingest.py" in result

    def test_extracts_full_path_mentions(self):
        answer = "Changes needed in tests/test_parser.py for the new field."
        result = extract_mentioned_files(answer, self.known_files)
        assert "tests/test_parser.py" in result

    def test_extracts_backtick_wrapped_files(self):
        answer = "Modify `parser.py` and `tools.py`."
        result = extract_mentioned_files(answer, self.known_files)
        assert "parser.py" in result
        assert "tools.py" in result

    def test_no_files_mentioned(self):
        answer = "This is a general discussion about code quality."
        result = extract_mentioned_files(answer, self.known_files)
        assert len(result) == 0

    def test_unknown_file_not_extracted(self):
        answer = "You should modify nonexistent_file.py"
        result = extract_mentioned_files(answer, self.known_files)
        assert "nonexistent_file.py" not in result

    def test_multiple_mentions_deduplicated(self):
        answer = "First update parser.py. Then check parser.py again."
        result = extract_mentioned_files(answer, self.known_files)
        assert len([f for f in result if f == "parser.py"]) <= 1


# ---------------------------------------------------------------------------
# Precision / Recall / F1
# ---------------------------------------------------------------------------

class TestScoreSingle:
    def test_perfect_score(self):
        mentioned = {"a.py", "b.py"}
        ground_truth = {"a.py", "b.py"}
        all_files = {"a.py", "b.py", "c.py"}
        p, r, f1, hallucinated = score_single(mentioned, ground_truth, all_files)
        assert p == 1.0
        assert r == 1.0
        assert f1 == 1.0
        assert hallucinated == set()

    def test_zero_recall(self):
        mentioned = set()
        ground_truth = {"a.py", "b.py"}
        all_files = {"a.py", "b.py", "c.py"}
        p, r, f1, _ = score_single(mentioned, ground_truth, all_files)
        assert p == 0.0
        assert r == 0.0
        assert f1 == 0.0

    def test_partial_recall(self):
        mentioned = {"a.py"}
        ground_truth = {"a.py", "b.py"}
        all_files = {"a.py", "b.py", "c.py"}
        p, r, f1, _ = score_single(mentioned, ground_truth, all_files)
        assert p == 1.0  # 1 correct out of 1 mentioned
        assert r == 0.5  # 1 correct out of 2 ground truth
        assert 0.6 < f1 < 0.7  # 2/3

    def test_false_positives_reduce_precision(self):
        mentioned = {"a.py", "c.py"}
        ground_truth = {"a.py"}
        all_files = {"a.py", "b.py", "c.py"}
        p, r, f1, _ = score_single(mentioned, ground_truth, all_files)
        assert p == 0.5  # 1 correct out of 2 mentioned
        assert r == 1.0  # 1 correct out of 1 ground truth

    def test_hallucinated_files_detected(self):
        mentioned = {"a.py", "hallucinated.py"}
        ground_truth = {"a.py"}
        all_files = {"a.py", "b.py"}
        _, _, _, hallucinated = score_single(mentioned, ground_truth, all_files)
        assert "hallucinated.py" in hallucinated

    def test_empty_ground_truth(self):
        mentioned = {"a.py"}
        ground_truth = set()
        all_files = {"a.py"}
        p, r, f1, _ = score_single(mentioned, ground_truth, all_files)
        assert r == 0.0
        assert f1 == 0.0


# ---------------------------------------------------------------------------
# Ground truth tasks validation
# ---------------------------------------------------------------------------

class TestGroundTruthTasks:
    def test_all_tasks_have_required_keys(self):
        for task in GROUND_TRUTH_TASKS:
            assert "prompt" in task
            assert "ground_truth_files" in task
            assert isinstance(task["ground_truth_files"], set)

    def test_all_tasks_have_nonempty_files(self):
        for task in GROUND_TRUTH_TASKS:
            assert len(task["ground_truth_files"]) > 0

    def test_ground_truth_files_are_python(self):
        for task in GROUND_TRUTH_TASKS:
            for f in task["ground_truth_files"]:
                assert f.endswith(".py"), f"Ground truth file {f} is not .py"

    def test_at_least_4_tasks(self):
        assert len(GROUND_TRUTH_TASKS) >= 4


# ---------------------------------------------------------------------------
# Data class tests
# ---------------------------------------------------------------------------

class TestTaskScore:
    def test_default_values(self):
        score = TaskScore(task_prompt="test", mode="Mode A")
        assert score.precision == 0.0
        assert score.recall == 0.0
        assert score.f1 == 0.0
        assert score.time_seconds == 0.0

    def test_sets_computed(self):
        score = TaskScore(
            task_prompt="test", mode="Mode A",
            mentioned_files={"a.py", "b.py"},
            ground_truth_files={"a.py", "c.py"},
            true_positives={"a.py"},
            false_positives={"b.py"},
            false_negatives={"c.py"},
        )
        assert len(score.true_positives) == 1
        assert len(score.false_negatives) == 1


class TestScoringReport:
    def test_default_empty(self):
        report = ScoringReport()
        assert report.task_scores == []
        assert report.mode_averages == {}
