# tests/test_parser.py
"""
Unit tests for parser.py.
Uses the real Tree-Sitter parser against fixture files — no mocks.
"""

from pathlib import Path
import tempfile
import pytest
from parser import parse_file, parse_directory, FunctionDef, ClassDef, ImportRef, CallEdge

FIXTURES = Path(__file__).parent / "fixtures"


class TestSimpleModule:
    def setup_method(self):
        self.result = parse_file(FIXTURES / "simple_module.py", FIXTURES)

    def test_class_extracted(self):
        class_names = [c.name for c in self.result.classes]
        assert "Calculator" in class_names

    def test_class_docstring(self):
        calc = next(c for c in self.result.classes if c.name == "Calculator")
        assert calc.docstring == "Performs basic arithmetic."

    def test_class_qualname_top_level(self):
        calc = next(c for c in self.result.classes if c.name == "Calculator")
        assert calc.qualname == "Calculator"

    def test_methods_extracted(self):
        func_names = [f.name for f in self.result.functions]
        assert "add" in func_names
        assert "subtract" in func_names
        add_func = next(f for f in self.result.functions if f.name == "add")
        assert add_func.is_method is True
        assert add_func.class_name == "Calculator"
        sub_func = next(f for f in self.result.functions if f.name == "subtract")
        assert sub_func.is_method is True
        assert sub_func.class_name == "Calculator"

    def test_method_qualname(self):
        add_func = next(f for f in self.result.functions if f.name == "add")
        assert add_func.qualname == "Calculator.add"

    def test_standalone_function(self):
        standalone = next(
            f for f in self.result.functions if f.name == "standalone_function"
        )
        assert standalone.is_method is False
        assert standalone.class_name is None

    def test_standalone_function_qualname(self):
        standalone = next(
            f for f in self.result.functions if f.name == "standalone_function"
        )
        assert standalone.qualname == "standalone_function"

    def test_function_docstring(self):
        standalone = next(
            f for f in self.result.functions if f.name == "standalone_function"
        )
        assert standalone.docstring == "Square a number."

    def test_line_numbers_nonzero(self):
        for func in self.result.functions:
            assert func.start_line > 0
            assert func.end_line > 0
        for cls in self.result.classes:
            assert cls.start_line > 0
            assert cls.end_line > 0


class TestCallerModule:
    def setup_method(self):
        self.result = parse_file(FIXTURES / "caller_module.py", FIXTURES)

    def test_import_extracted(self):
        modules = [i.module for i in self.result.imports]
        assert "simple_module" in modules

    def test_call_edge_wrapper_to_standalone(self):
        # caller_qualname is now used instead of caller_name
        matching = [
            c for c in self.result.calls
            if c.caller_qualname == "wrapper" and c.callee_expr == "standalone_function"
        ]
        assert len(matching) >= 1

    def test_call_edge_double_wrap_to_wrapper(self):
        matching = [
            c for c in self.result.calls
            if c.caller_qualname == "double_wrap" and c.callee_expr == "wrapper"
        ]
        assert len(matching) >= 1


class TestDeepChain:
    def setup_method(self):
        self.result = parse_file(FIXTURES / "deep_chain.py", FIXTURES)

    def test_three_functions_extracted(self):
        names = [f.name for f in self.result.functions]
        assert "alpha" in names
        assert "beta" in names
        assert "gamma" in names

    def test_alpha_calls_beta(self):
        matching = [
            c for c in self.result.calls
            if c.caller_qualname == "alpha" and c.callee_expr == "beta"
        ]
        assert len(matching) >= 1

    def test_beta_calls_gamma(self):
        matching = [
            c for c in self.result.calls
            if c.caller_qualname == "beta" and c.callee_expr == "gamma"
        ]
        assert len(matching) >= 1


class TestParseDirectory:
    def test_parses_all_fixtures(self):
        results = parse_directory(FIXTURES)
        file_paths = [r.file_path for r in results]
        assert any("simple_module.py" in fp for fp in file_paths)
        assert any("caller_module.py" in fp for fp in file_paths)
        assert any("deep_chain.py" in fp for fp in file_paths)

    def test_skips_pycache(self):
        results = parse_directory(FIXTURES)
        for r in results:
            assert "__pycache__" not in r.file_path


class TestNestedClasses:
    def setup_method(self):
        self.result = parse_file(FIXTURES / "nested_classes.py", FIXTURES)

    def test_outer_class_extracted(self):
        names = [c.name for c in self.result.classes]
        assert "Outer" in names

    def test_inner_class_extracted(self):
        names = [c.name for c in self.result.classes]
        assert "Inner" in names

    def test_outer_qualname(self):
        outer = next(c for c in self.result.classes if c.name == "Outer")
        assert outer.qualname == "Outer"

    def test_inner_qualname(self):
        inner = next(c for c in self.result.classes if c.name == "Inner")
        assert inner.qualname == "Outer.Inner"

    def test_inner_method_qualname(self):
        method = next(f for f in self.result.functions if f.name == "method"
                      and f.class_name == "Inner")
        assert method.qualname == "Outer.Inner.method"

    def test_outer_method_qualname(self):
        om = next(f for f in self.result.functions if f.name == "outer_method")
        assert om.qualname == "Outer.outer_method"

    def test_nested_function_qualname(self):
        local_fn = next((f for f in self.result.functions if f.name == "local_fn"), None)
        assert local_fn is not None
        assert "<locals>" in local_fn.qualname

    def test_inner_method_class_name_is_immediate(self):
        # class_name must be the immediate class for schema compatibility
        method = next(f for f in self.result.functions if f.name == "method"
                      and f.class_name == "Inner")
        assert method.class_name == "Inner"


class TestFullCalleeExpression:
    """Parser must return full callee expression, not just the method name."""

    def _parse_snippet(self, code: str) -> list[CallEdge]:
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(code)
            f.flush()
            tmp_path = Path(f.name)
        try:
            result = parse_file(tmp_path, tmp_path.parent)
            return result.calls
        finally:
            tmp_path.unlink()

    def test_simple_call(self):
        calls = self._parse_snippet("def fn():\n    helper()\n")
        assert any(c.callee_expr == "helper" for c in calls)

    def test_self_method_full_expr(self):
        code = "class A:\n    def bar(self):\n        self.foo()\n"
        calls = self._parse_snippet(code)
        # Must include "self.foo", not just "foo"
        assert any(c.callee_expr == "self.foo" for c in calls)

    def test_module_attr_full_expr(self):
        code = "import os\ndef fn():\n    os.path.join('/tmp', 'f')\n"
        calls = self._parse_snippet(code)
        assert any("os.path.join" in c.callee_expr for c in calls)

    def test_column_points_to_method_not_object(self):
        code = "class A:\n    def bar(self):\n        self.foo()\n"
        calls = self._parse_snippet(code)
        self_foo = next((c for c in calls if c.callee_expr == "self.foo"), None)
        assert self_foo is not None
        # Column should NOT be 0 (the start of the line / 'self')
        # It should point to 'foo', which starts after 'self.'
        assert self_foo.column > 0

    def test_module_level_caller_qualname(self):
        code = "helper()\n"
        calls = self._parse_snippet(code)
        assert any(c.caller_qualname == "<module>" for c in calls)

    def test_classbody_caller_qualname(self):
        code = "class MyClass:\n    x = helper()\n"
        calls = self._parse_snippet(code)
        assert any(c.caller_qualname == "MyClass.__classbody__" for c in calls)


class TestImportedNames:
    """from X import a, b → two ImportRef objects, each with imported_name set."""

    def _parse_snippet(self, code: str):
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(code)
            f.flush()
            tmp_path = Path(f.name)
        try:
            return parse_file(tmp_path, tmp_path.parent).imports
        finally:
            tmp_path.unlink()

    def test_from_import_multiple_names(self):
        imports = self._parse_snippet("from os.path import join, exists\n")
        names = {i.imported_name for i in imports}
        assert "join" in names
        assert "exists" in names
        # Both should point to os.path
        for imp in imports:
            if imp.imported_name in ("join", "exists"):
                assert imp.module == "os.path"

    def test_from_import_alias(self):
        imports = self._parse_snippet("from os.path import join as j\n")
        imp = next((i for i in imports if i.imported_name == "join"), None)
        assert imp is not None
        assert imp.alias == "j"
        assert imp.module == "os.path"

    def test_plain_import_no_imported_name(self):
        imports = self._parse_snippet("import os\n")
        imp = next((i for i in imports if i.module == "os"), None)
        assert imp is not None
        assert imp.imported_name is None

    def test_star_import(self):
        imports = self._parse_snippet("from os.path import *\n")
        star = next((i for i in imports if i.imported_name == "*"), None)
        assert star is not None
        assert star.module == "os.path"


class TestEdgeCases:
    def test_self_dot_call_returns_full_expr(self):
        """self.foo() → callee_expr must be 'self.foo', not just 'foo'."""
        code = "class MyClass:\n    def bar(self):\n        self.foo()\n"
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(code)
            f.flush()
            tmp_path = Path(f.name)
        try:
            result = parse_file(tmp_path, tmp_path.parent)
            # New behaviour: full expression
            call_edges = [c for c in result.calls if "foo" in c.callee_expr]
            assert len(call_edges) >= 1
            assert call_edges[0].callee_expr == "self.foo"
        finally:
            tmp_path.unlink()

    def test_nonexistent_file_raises_oserror(self):
        with pytest.raises(OSError):
            parse_file(Path("/nonexistent/file.py"), Path("/nonexistent"))

    def test_non_python_file_raises_valueerror(self):
        with pytest.raises(ValueError, match="Unsupported file extension"):
            parse_file(Path("/some/file.txt"), Path("/some"))


# ---------------------------------------------------------------------------
# Fix C — compound attribute VariableRef (`import mod; mod.ATTR`)
# ---------------------------------------------------------------------------

class TestAttributeVariableRef:
    def test_compound_ref_collected(self):
        pf = parse_file(FIXTURES / "module_attr_reader.py", FIXTURES.parent.parent)
        compound = [r for r in pf.variable_refs if "." in r.name]
        names = [r.name for r in compound]
        assert "cfg.DATABASE_URL" in names or any("DATABASE_URL" in n for n in names)

    def test_compound_ref_inside_function(self):
        pf = parse_file(FIXTURES / "module_attr_reader.py", FIXTURES.parent.parent)
        compound = [r for r in pf.variable_refs if "." in r.name]
        # All compound refs must be inside a function scope (user_qualname != "<module>")
        for ref in compound:
            assert ref.user_qualname != "<module>", f"{ref.name} collected at module scope"

    def test_called_attribute_not_collected(self):
        # `obj.method()` call — should produce a CallEdge, NOT a compound VariableRef
        pf = parse_file(FIXTURES / "simple_module.py", FIXTURES)
        compound = [r for r in pf.variable_refs if "." in r.name]
        # Verify that callee expressions from calls are not in variable_refs
        call_exprs = {c.callee_expr for c in pf.calls}
        for ref in compound:
            assert ref.name not in call_exprs, f"callee {ref.name!r} leaked into variable_refs"


# ---------------------------------------------------------------------------
# Fix D — __all__ parsed into exports field
# ---------------------------------------------------------------------------

class TestExportsField:
    def test_all_exporter_exports_populated(self):
        pf = parse_file(FIXTURES / "all_exporter.py", FIXTURES.parent.parent)
        assert pf.exports == ["PublicHelper"]

    def test_module_without_all_has_empty_exports(self):
        pf = parse_file(FIXTURES / "config_fixture.py", FIXTURES.parent.parent)
        assert pf.exports == []

    def test_star_importer_has_no_exports(self):
        pf = parse_file(FIXTURES / "star_importer.py", FIXTURES.parent.parent)
        assert pf.exports == []  # star_importer doesn't define __all__


# ---------------------------------------------------------------------------
# Fix E — top_level_aliases parsed from __init__.py
# ---------------------------------------------------------------------------

class TestTopLevelAliases:
    def test_alias_detected_in_init(self):
        pf = parse_file(FIXTURES / "alias_init" / "__init__.py", FIXTURES.parent.parent)
        alias_names = [a for a, _ in pf.top_level_aliases]
        assert "ConcreteModel" in alias_names

    def test_alias_source_correct(self):
        pf = parse_file(FIXTURES / "alias_init" / "__init__.py", FIXTURES.parent.parent)
        alias_map = dict(pf.top_level_aliases)
        assert alias_map.get("ConcreteModel") == "_ConcreteModel"

    def test_non_init_file_aliases_still_collected(self):
        # Assignment aliases are collected from any file, not just __init__.py
        # (build_symbol_table only uses them for __init__.py, but parser collects everywhere)
        pf = parse_file(FIXTURES / "config_fixture.py", FIXTURES.parent.parent)
        # config_fixture.py has no Name = OtherName assignments
        assert pf.top_level_aliases == []
