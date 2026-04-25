# parser.py
"""
Parse Python files using Tree-Sitter and return structured AST entities.
Does NOT touch FalkorDB.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import tree_sitter_python as tspython
from tree_sitter import Language, Parser

PY_LANGUAGE = Language(tspython.language())

SKIP_DIRS = {"__pycache__", ".git", ".venv", "node_modules"}


@dataclass
class FunctionDef:
    name: str
    file_path: str          # Relative to ingested repo root
    start_line: int
    end_line: int
    docstring: Optional[str]
    is_method: bool         # True if defined inside a class body
    class_name: Optional[str]  # Populated when is_method=True


@dataclass
class ClassDef:
    name: str
    file_path: str
    start_line: int
    end_line: int
    docstring: Optional[str]
    bases: list[str] = field(default_factory=list)


@dataclass
class ImportRef:
    file_path: str
    module: str             # e.g. "os.path" or "numpy"
    alias: Optional[str]    # e.g. "np" in "import numpy as np"


@dataclass
class CallEdge:
    caller_name: str        # Fully qualified: ClassName.method_name or function_name
    callee_name: str        # Raw name as it appears at the call site
    file_path: str
    line: int


@dataclass
class ParsedFile:
    file_path: str
    functions: list[FunctionDef] = field(default_factory=list)
    classes: list[ClassDef] = field(default_factory=list)
    imports: list[ImportRef] = field(default_factory=list)
    calls: list[CallEdge] = field(default_factory=list)


def _extract_docstring(node) -> Optional[str]:
    """
    Extract a docstring from the body of a function_definition or class_definition.
    The docstring is the first expression_statement child whose child is a string node,
    appearing immediately after the colon block.
    """
    body = node.child_by_field_name("body")
    if body is None:
        return None

    for child in body.children:
        # Skip newlines and indentation
        if child.type in ("newline", "indent", "dedent", "comment"):
            continue
        if child.type == "expression_statement":
            string_node = child.children[0] if child.children else None
            if string_node and string_node.type == "string":
                raw = string_node.text.decode("utf-8")
                # Strip triple-quote delimiters and whitespace
                for delim in ('"""', "'''"):
                    if raw.startswith(delim) and raw.endswith(delim):
                        raw = raw[3:-3]
                        break
                else:
                    # Single-quote string used as docstring
                    if raw.startswith('"') and raw.endswith('"'):
                        raw = raw[1:-1]
                    elif raw.startswith("'") and raw.endswith("'"):
                        raw = raw[1:-1]
                return raw.strip()
            break  # First non-trivial statement is not a docstring
        else:
            break  # First non-trivial statement is not an expression_statement
    return None


def _find_enclosing_class(node) -> Optional[str]:
    """Walk up the parent chain to find if inside a class_definition. Return class name or None."""
    current = node.parent
    while current is not None:
        if current.type == "class_definition":
            name_node = current.child_by_field_name("name")
            if name_node:
                return name_node.text.decode("utf-8")
        if current.type == "module":
            break
        current = current.parent
    return None


def _find_enclosing_function(node) -> Optional[str]:
    """Walk up parent chain to find the enclosing function_definition. Return its name or None."""
    current = node.parent
    while current is not None:
        if current.type == "function_definition":
            name_node = current.child_by_field_name("name")
            if name_node:
                return name_node.text.decode("utf-8")
        if current.type == "module":
            break
        current = current.parent
    return None


def _resolve_callee_name(call_node) -> Optional[str]:
    """
    Resolve the callee name from a call node.
    Handles: foo(), self.foo(), obj.method()
    Returns just the method/function name.
    """
    func = call_node.child_by_field_name("function")
    if func is None:
        return None

    if func.type == "identifier":
        return func.text.decode("utf-8")
    elif func.type == "attribute":
        # obj.method() or self.method() → return "method"
        attr = func.child_by_field_name("attribute")
        if attr:
            return attr.text.decode("utf-8")
    return None


def _get_caller_context(node, file_path: str) -> str:
    """
    Determine the caller context for a call node.
    If inside a function, return "ClassName.method_name" or "function_name".
    If inside a class body but outside any function, return "ClassName.__classbody__".
    """
    # First, look for enclosing function
    enclosing_func = _find_enclosing_function(node)
    enclosing_class = _find_enclosing_class(node)

    if enclosing_func is not None:
        if enclosing_class is not None:
            return f"{enclosing_class}.{enclosing_func}"
        return enclosing_func
    elif enclosing_class is not None:
        return f"{enclosing_class}.__classbody__"
    else:
        return "<module>"


def _extract_import_module(node) -> list[tuple[str, Optional[str]]]:
    """
    Extract (module, alias) tuples from import_statement or import_from_statement nodes.
    """
    results = []

    if node.type == "import_statement":
        # import foo / import foo as f / import foo, bar
        for child in node.children:
            if child.type == "dotted_name":
                module = child.text.decode("utf-8")
                results.append((module, None))
            elif child.type == "aliased_import":
                name_node = child.child_by_field_name("name")
                alias_node = child.child_by_field_name("alias")
                if name_node:
                    module = name_node.text.decode("utf-8")
                    alias = alias_node.text.decode("utf-8") if alias_node else None
                    results.append((module, alias))

    elif node.type == "import_from_statement":
        # from foo.bar import baz / from foo.bar import baz as b
        module_node = node.child_by_field_name("module_name")
        if module_node:
            module = module_node.text.decode("utf-8")
        else:
            # Try to find the dotted_name manually
            module = None
            for child in node.children:
                if child.type == "dotted_name" or child.type == "relative_import":
                    module = child.text.decode("utf-8")
                    break
            if module is None:
                return results

        # Check for aliases among imported names
        alias = None
        for child in node.children:
            if child.type == "aliased_import":
                alias_node = child.child_by_field_name("alias")
                if alias_node:
                    alias = alias_node.text.decode("utf-8")
                break

        results.append((module, alias))

    return results


def _walk_tree(node, result: ParsedFile, source_bytes: bytes, rel_path: str) -> None:
    """Recursively walk the AST and extract entities."""

    if node.type == "class_definition":
        name_node = node.child_by_field_name("name")
        superclasses_node = node.child_by_field_name("superclasses")
        bases = []
        if superclasses_node:
            for child in superclasses_node.children:
                if child.type == "identifier" or child.type == "attribute":
                    bases.append(child.text.decode("utf-8"))
        if name_node:
            class_name = name_node.text.decode("utf-8")
            docstring = _extract_docstring(node)
            result.classes.append(ClassDef(
                name=class_name,
                file_path=rel_path,
                start_line=node.start_point[0] + 1,  # Tree-sitter is 0-indexed
                end_line=node.end_point[0] + 1,
                docstring=docstring,
                bases=bases,
            ))

    elif node.type == "function_definition":
        name_node = node.child_by_field_name("name")
        if name_node:
            func_name = name_node.text.decode("utf-8")
            docstring = _extract_docstring(node)
            enclosing_class = _find_enclosing_class(node)
            result.functions.append(FunctionDef(
                name=func_name,
                file_path=rel_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                docstring=docstring,
                is_method=enclosing_class is not None,
                class_name=enclosing_class,
            ))

    elif node.type in ("import_statement", "import_from_statement"):
        imports = _extract_import_module(node)
        for module, alias in imports:
            result.imports.append(ImportRef(
                file_path=rel_path,
                module=module,
                alias=alias,
            ))

    elif node.type == "call":
        callee_name = _resolve_callee_name(node)
        if callee_name:
            caller_name = _get_caller_context(node, rel_path)
            result.calls.append(CallEdge(
                caller_name=caller_name,
                callee_name=callee_name,
                file_path=rel_path,
                line=node.start_point[0] + 1,
            ))

    # Recurse into children
    for child in node.children:
        _walk_tree(child, result, source_bytes, rel_path)


def parse_file(file_path: Path, repo_root: Path) -> ParsedFile:
    """
    Parse a single Python file and return all extracted entities.

    Args:
        file_path: Absolute path to the .py file.
        repo_root: Absolute path to the repo root (used to compute relative paths).

    Returns:
        ParsedFile with all functions, classes, imports, and call edges found.

    Raises:
        ValueError: If file_path does not end in .py.
        OSError: If the file cannot be read.
    """
    file_path = Path(file_path)
    repo_root = Path(repo_root)

    if not str(file_path).endswith(".py"):
        raise ValueError(f"Expected a .py file, got: {file_path}")

    source_bytes = file_path.read_bytes()

    rel_path = str(file_path.relative_to(repo_root))

    parser = Parser(PY_LANGUAGE)
    tree = parser.parse(source_bytes)

    result = ParsedFile(file_path=rel_path)
    _walk_tree(tree.root_node, result, source_bytes, rel_path)

    return result


def parse_directory(repo_root: Path) -> list[ParsedFile]:
    """
    Walk repo_root recursively, parse all .py files, and return results.
    Skips files inside __pycache__, .git, .venv, and node_modules directories.

    Args:
        repo_root: Absolute path to the root of the Python project.

    Returns:
        List of ParsedFile, one per .py file found.
    """
    repo_root = Path(repo_root).resolve()
    results = []

    for py_file in sorted(repo_root.rglob("*.py")):
        # Check if any parent directory is in SKIP_DIRS
        parts = py_file.relative_to(repo_root).parts
        if any(part in SKIP_DIRS for part in parts):
            continue

        try:
            parsed = parse_file(py_file, repo_root)
            results.append(parsed)
        except (ValueError, OSError):
            # Skip files that can't be parsed
            continue

    return results
