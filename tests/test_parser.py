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

    def test_standalone_function(self):
        standalone = next(
            f for f in self.result.functions if f.name == "standalone_function"
        )
        assert standalone.is_method is False
        assert standalone.class_name is None

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
        matching = [
            c for c in self.result.calls
            if c.caller_name == "wrapper" and c.callee_name == "standalone_function"
        ]
        assert len(matching) >= 1

    def test_call_edge_double_wrap_to_wrapper(self):
        matching = [
            c for c in self.result.calls
            if c.caller_name == "double_wrap" and c.callee_name == "wrapper"
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
            if c.caller_name == "alpha" and c.callee_name == "beta"
        ]
        assert len(matching) >= 1

    def test_beta_calls_gamma(self):
        matching = [
            c for c in self.result.calls
            if c.caller_name == "beta" and c.callee_name == "gamma"
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


class TestEdgeCases:
    def test_self_dot_call_resolves_to_method_name(self):
        """Create a temp .py file with self.foo() inside a method,
        assert resulting CallEdge has callee_name='foo'."""
        code = '''
class MyClass:
    def bar(self):
        self.foo()
'''
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(code)
            f.flush()
            tmp_path = Path(f.name)

        try:
            result = parse_file(tmp_path, tmp_path.parent)
            call_edges = [c for c in result.calls if c.callee_name == "foo"]
            assert len(call_edges) >= 1
            assert call_edges[0].callee_name == "foo"
        finally:
            tmp_path.unlink()

    def test_nonexistent_file_raises_oserror(self):
        with pytest.raises(OSError):
            parse_file(Path("/nonexistent/file.py"), Path("/nonexistent"))

    def test_non_python_file_raises_valueerror(self):
        with pytest.raises(ValueError, match="Unsupported file extension"):
            parse_file(Path("/some/file.txt"), Path("/some"))
