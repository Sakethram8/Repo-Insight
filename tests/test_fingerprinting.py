"""Unit tests for fingerprinting.py — Tier 0 (static) and Tier 1 (skeleton)."""

import pytest
from fingerprinting import build_static_fingerprint, build_code_skeleton, inject_behavior_label


# ---------------------------------------------------------------------------
# build_static_fingerprint
# ---------------------------------------------------------------------------

class TestBuildStaticFingerprint:
    def test_basic_signature(self):
        fp = build_static_fingerprint(
            name="login", qualname="auth.User.login",
            params=["self", "password: str"], return_annotation="bool",
            docstring=None, raises=[], calls=[], reads=[], caller_count=0,
        )
        assert fp.startswith("login(self, password: str) → bool")

    def test_return_annotation_omitted_when_none(self):
        fp = build_static_fingerprint(
            name="cleanup", qualname="cleanup",
            params=[], return_annotation=None,
            docstring=None, raises=[], calls=[], reads=[], caller_count=0,
        )
        assert "→" not in fp.split("\n")[0]

    def test_docstring_used_as_behavior_hint(self):
        fp = build_static_fingerprint(
            name="hash_pw", qualname="hash_pw",
            params=["pw: str"], return_annotation="str",
            docstring="Hash a password using bcrypt. Returns digest.",
            raises=[], calls=[], reads=[], caller_count=3,
        )
        assert "// Hash a password using bcrypt" in fp

    def test_behavior_label_overrides_docstring(self):
        fp = build_static_fingerprint(
            name="foo", qualname="foo",
            params=[], return_annotation=None,
            docstring="Old doc",
            raises=[], calls=[], reads=[], caller_count=0,
            behavior_label="Validates input and raises on bad token",
        )
        assert "// Validates input and raises on bad token" in fp
        assert "Old doc" not in fp

    def test_calls_listed(self):
        fp = build_static_fingerprint(
            name="process", qualname="process",
            params=[], return_annotation=None,
            docstring=None, raises=[],
            calls=["db.save", "cache.set", "notify.send"],
            reads=[], caller_count=0,
        )
        assert "calls: save, set, send" in fp

    def test_calls_truncated_at_six(self):
        calls = [f"mod.fn{i}" for i in range(10)]
        fp = build_static_fingerprint(
            name="big", qualname="big",
            params=[], return_annotation=None,
            docstring=None, raises=[], calls=calls, reads=[], caller_count=0,
        )
        assert "…+4" in fp

    def test_reads_and_raises_listed(self):
        fp = build_static_fingerprint(
            name="load", qualname="load",
            params=[], return_annotation=None,
            docstring=None,
            raises=["FileNotFoundError", "PermissionError"],
            calls=[],
            reads=["config.PATH", "env.HOME"],
            caller_count=5,
        )
        assert "reads: config.PATH, env.HOME" in fp
        assert "raises: FileNotFoundError, PermissionError" in fp
        assert "callers: 5" in fp

    def test_caller_count_zero(self):
        fp = build_static_fingerprint(
            name="leaf", qualname="leaf",
            params=[], return_annotation=None,
            docstring=None, raises=[], calls=[], reads=[], caller_count=0,
        )
        assert "callers: 0" in fp


# ---------------------------------------------------------------------------
# build_code_skeleton
# ---------------------------------------------------------------------------

class TestBuildCodeSkeleton:
    def test_removes_long_string_literals(self):
        src = """
def greet(name):
    msg = "Hello, this is a very long string that should be replaced"
    return msg
"""
        skel = build_code_skeleton(src)
        assert "Hello, this is a very long string" not in skel

    def test_keeps_short_string_literals(self):
        src = """
def mode(flag):
    if flag == "on":
        return True
    return False
"""
        skel = build_code_skeleton(src)
        # ast.unparse normalises to single quotes; both forms are acceptable
        assert "'on'" in skel or '"on"' in skel

    def test_drops_call_arguments(self):
        # When a call is the RHS of an assignment, visit_Assign replaces the whole
        # RHS with ..., so the call disappears. For standalone calls inside if/return,
        # arguments are stripped.
        src = """
def process(data):
    if transform(data, key="value", reverse=True):
        return True
    return False
"""
        skel = build_code_skeleton(src)
        assert "transform()" in skel

    def test_keeps_control_flow_conditions(self):
        src = """
def classify(x):
    if x > 100:
        return "large"
    elif x > 10:
        return "medium"
    return "small"
"""
        skel = build_code_skeleton(src)
        assert "x > 100" in skel
        assert "x > 10" in skel

    def test_keeps_return_statements(self):
        src = """
def compute(n):
    if n < 0:
        return -1
    return n * 2
"""
        skel = build_code_skeleton(src)
        assert "return" in skel

    def test_keeps_raise_statements(self):
        src = """
def validate(val):
    if val is None:
        raise ValueError("Cannot be None")
    return val
"""
        skel = build_code_skeleton(src)
        assert "raise ValueError" in skel

    def test_drops_import_statements(self):
        src = """
def load():
    import os
    from pathlib import Path
    return os.getcwd()
"""
        skel = build_code_skeleton(src)
        assert "import" not in skel

    def test_compression_below_70_percent(self):
        src = """
def _write_call_and_reads_edges(parsed_files, graph, module_fqn_map, reexport_map, all_names):
    pending_calls = []
    for pf in parsed_files:
        for func in pf.functions:
            func_fqn = module_fqn_map.get(pf.module_name, pf.module_name) + "." + func.name
            for call_name in func.calls:
                resolved = module_fqn_map.get(call_name)
                if resolved:
                    graph.query("MATCH (a:Function {fqn: $a}), (b:Function {fqn: $b}) MERGE (a)-[:CALLS]->(b)", {"a": func_fqn, "b": resolved})
                else:
                    pending_calls.append((func_fqn, call_name, "A very long message explaining why this is pending due to complex resolution"))
    return pending_calls
"""
        skel = build_code_skeleton(src)
        ratio = len(skel) / len(src)
        assert ratio < 0.80, f"Expected <80% compression, got {ratio:.0%}: {skel}"

    def test_syntax_error_returns_original(self):
        src = "def broken(\n  # unterminated"
        skel = build_code_skeleton(src)
        assert skel == src


# ---------------------------------------------------------------------------
# inject_behavior_label
# ---------------------------------------------------------------------------

class TestInjectBehaviorLabel:
    def test_inserts_label_into_fingerprint_without_label(self):
        fp = "foo(x: int) → bool\ncalls: bar\ncallers: 3"
        result = inject_behavior_label(fp, "Checks if x is valid")
        lines = result.split("\n")
        assert lines[0] == "foo(x: int) → bool"
        assert lines[1] == "// Checks if x is valid"
        assert "calls: bar" in result

    def test_replaces_existing_label(self):
        fp = "foo(x: int) → bool\n// Old label\ncalls: bar\ncallers: 3"
        result = inject_behavior_label(fp, "New label")
        assert "// New label" in result
        assert "// Old label" not in result

    def test_strips_leading_slashes_from_label(self):
        fp = "fn() → None\ncallers: 0"
        result = inject_behavior_label(fp, "// Handles cleanup")
        assert "// Handles cleanup" in result
        assert "// // " not in result

    def test_label_truncated_at_120_chars(self):
        label = "x" * 200
        fp = "fn() → None\ncallers: 0"
        result = inject_behavior_label(fp, label)
        label_line = [l for l in result.split("\n") if l.startswith("//")][0]
        assert len(label_line) <= 123  # "// " + 120
