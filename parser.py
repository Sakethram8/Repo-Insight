# parser.py
"""
Parse source files using Tree-Sitter and return structured AST entities.
Supports Python, JavaScript, and TypeScript.
Does NOT touch FalkorDB.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import tree_sitter_python as tspython
from tree_sitter import Language, Parser

logger = logging.getLogger(__name__)

PY_LANGUAGE = Language(tspython.language())

# Optional JS/TS support — graceful degradation if not installed
try:
    import tree_sitter_javascript as tsjavascript
    JS_LANGUAGE = Language(tsjavascript.language())
except ImportError:
    JS_LANGUAGE = None
    logger.info("tree-sitter-javascript not installed; JS parsing disabled")

try:
    import tree_sitter_typescript as tstypescript
    TS_LANGUAGE = Language(tstypescript.language_typescript())
    TSX_LANGUAGE = Language(tstypescript.language_tsx())
except ImportError:
    TS_LANGUAGE = None
    TSX_LANGUAGE = None
    logger.info("tree-sitter-typescript not installed; TS parsing disabled")

# Map file extensions to languages
_EXTENSION_MAP: dict[str, Language | None] = {
    ".py": PY_LANGUAGE,
    ".js": JS_LANGUAGE,
    ".jsx": JS_LANGUAGE,
    ".ts": TS_LANGUAGE,
    ".tsx": TSX_LANGUAGE,
}

SUPPORTED_EXTENSIONS = {ext for ext, lang in _EXTENSION_MAP.items() if lang is not None}

from config import SKIP_DIRS
SKIP_DIRS = set(SKIP_DIRS)


@dataclass
class FunctionDef:
    name: str
    file_path: str          # Relative to ingested repo root
    start_line: int
    end_line: int
    docstring: Optional[str]
    is_method: bool         # True if defined inside a class body
    class_name: Optional[str]  # Populated when is_method=True
    params: list[str] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)
    return_annotation: Optional[str] = None


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
    column: int = 0


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


def _resolve_relative_import(module: str, source_file_path: str) -> str:
    """Convert a relative import string to an absolute dotted module path.

    Examples:
        _resolve_relative_import(".utils", "mypackage/parser.py") -> "mypackage.utils"
        _resolve_relative_import(".", "mypackage/sub/file.py")    -> "mypackage.sub"
        _resolve_relative_import("os", "mypackage/file.py")       -> "os"  (unchanged)
    """
    if not module.startswith("."):
        return module
    level = len(module) - len(module.lstrip("."))
    rel_part = module.lstrip(".")
    # Convert file path to package parts
    parts = source_file_path.replace("\\", "/").replace(".py", "").split("/")
    # Go up `level` directories from the file's package
    base_parts = parts[:-level] if level <= len(parts) else []
    base = ".".join(p for p in base_parts if p and p != ".")
    if rel_part:
        return f"{base}.{rel_part}" if base else rel_part
    return base if base else "."


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

            # Extract parameters
            params = []
            params_node = node.child_by_field_name("parameters")
            if params_node:
                for child in params_node.children:
                    if child.type == "identifier":
                        params.append(child.text.decode("utf-8"))
                    elif child.type in ("typed_parameter", "default_parameter"):
                        pname = child.child_by_field_name("name")
                        if pname is None and child.children:
                            pname = child.children[0]
                        if pname:
                            params.append(pname.text.decode("utf-8"))

            # Extract decorators
            decorators = []
            sibling = node.prev_sibling
            while sibling and sibling.type == "decorator":
                decorators.append(sibling.text.decode("utf-8").lstrip("@").strip())
                sibling = sibling.prev_sibling

            # Extract return annotation
            return_annotation = None
            return_type_node = node.child_by_field_name("return_type")
            if return_type_node:
                return_annotation = return_type_node.text.decode("utf-8").lstrip("->").strip()

            result.functions.append(FunctionDef(
                name=func_name,
                file_path=rel_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                docstring=docstring,
                is_method=enclosing_class is not None,
                class_name=enclosing_class,
                params=params,
                decorators=decorators,
                return_annotation=return_annotation,
            ))

    elif node.type in ("import_statement", "import_from_statement"):
        imports = _extract_import_module(node)
        for module, alias in imports:
            module = _resolve_relative_import(module, rel_path)
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
                column=node.start_point[1],
            ))

    # Recurse into children
    for child in node.children:
        _walk_tree(child, result, source_bytes, rel_path)

# ---------------------------------------------------------------------------
# JavaScript / TypeScript AST walking
# ---------------------------------------------------------------------------

def _extract_js_docstring(node) -> Optional[str]:
    """Extract a JSDoc comment from immediately before a function/class node."""
    prev = node.prev_sibling
    if prev and prev.type == "comment":
        text = prev.text.decode("utf-8").strip()
        # Strip /** ... */ delimiters
        if text.startswith("/**") and text.endswith("*/"):
            text = text[3:-2].strip()
        elif text.startswith("//"):
            text = text[2:].strip()
        return text
    return None


def _walk_js_tree(node, result: ParsedFile, source_bytes: bytes, rel_path: str,
                  enclosing_class: Optional[str] = None) -> None:
    """Recursively walk a JS/TS AST and extract entities."""

    if node.type in ("class_declaration", "class"):
        name_node = node.child_by_field_name("name")
        if name_node:
            class_name = name_node.text.decode("utf-8")
            docstring = _extract_js_docstring(node)
            # Extract superclass
            bases = []
            heritage = node.child_by_field_name("heritage")
            if heritage is None:
                # Try to find class_heritage child
                for child in node.children:
                    if child.type == "class_heritage":
                        heritage = child
                        break
            if heritage:
                for child in heritage.children:
                    if child.type == "identifier":
                        bases.append(child.text.decode("utf-8"))

            result.classes.append(ClassDef(
                name=class_name,
                file_path=rel_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                docstring=docstring,
                bases=bases,
            ))
            # Recurse into class body with class context
            body = node.child_by_field_name("body")
            if body:
                for child in body.children:
                    _walk_js_tree(child, result, source_bytes, rel_path, enclosing_class=class_name)
            return  # Don't recurse again below

    elif node.type in ("function_declaration", "method_definition",
                       "arrow_function", "function"):
        name_node = node.child_by_field_name("name")
        func_name = None
        if name_node:
            func_name = name_node.text.decode("utf-8")
        elif node.type == "arrow_function" and node.parent:
            # Check if assigned to a variable: const foo = () => ...
            if node.parent.type == "variable_declarator":
                var_name = node.parent.child_by_field_name("name")
                if var_name:
                    func_name = var_name.text.decode("utf-8")

        if func_name:
            docstring = _extract_js_docstring(node)
            is_method = enclosing_class is not None or node.type == "method_definition"
            result.functions.append(FunctionDef(
                name=func_name,
                file_path=rel_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                docstring=docstring,
                is_method=is_method,
                class_name=enclosing_class,
            ))

    elif node.type in ("import_statement", "import_declaration"):
        # Extract module from: import ... from 'module'
        source_node = node.child_by_field_name("source")
        if source_node:
            module = source_node.text.decode("utf-8").strip("'\"")
            module = _resolve_relative_import(module, rel_path)
            result.imports.append(ImportRef(
                file_path=rel_path,
                module=module,
                alias=None,
            ))

    elif node.type == "call_expression":
        func = node.child_by_field_name("function")
        callee_name = None
        if func:
            if func.type == "identifier":
                callee_name = func.text.decode("utf-8")
            elif func.type == "member_expression":
                prop = func.child_by_field_name("property")
                if prop:
                    callee_name = prop.text.decode("utf-8")

        if callee_name:
            # Determine caller context
            caller_name = _find_js_enclosing_function(node) or "<module>"
            if enclosing_class and caller_name != "<module>":
                caller_name = f"{enclosing_class}.{caller_name}"

            if caller_name != "<module>":
                result.calls.append(CallEdge(
                    caller_name=caller_name,
                    callee_name=callee_name,
                    file_path=rel_path,
                    line=node.start_point[0] + 1,
                    column=node.start_point[1],
                ))

    # Recurse into children (skip if we already recursed for class body)
    for child in node.children:
        _walk_js_tree(child, result, source_bytes, rel_path, enclosing_class)


def _find_js_enclosing_function(node) -> Optional[str]:
    """Walk up parent chain to find enclosing function in JS/TS."""
    current = node.parent
    while current is not None:
        if current.type in ("function_declaration", "method_definition",
                            "arrow_function", "function"):
            name_node = current.child_by_field_name("name")
            if name_node:
                return name_node.text.decode("utf-8")
            # Arrow function assigned to variable
            if current.type == "arrow_function" and current.parent:
                if current.parent.type == "variable_declarator":
                    var_name = current.parent.child_by_field_name("name")
                    if var_name:
                        return var_name.text.decode("utf-8")
        if current.type in ("program", "module"):
            break
        current = current.parent
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_file(file_path: Path, repo_root: Path) -> ParsedFile:
    """
    Parse a single source file and return all extracted entities.

    Supports: .py, .js, .jsx, .ts, .tsx

    Args:
        file_path: Absolute path to the source file.
        repo_root: Absolute path to the repo root (used to compute relative paths).

    Returns:
        ParsedFile with all functions, classes, imports, and call edges found.

    Raises:
        ValueError: If file extension is not supported.
        OSError: If the file cannot be read.
    """
    file_path = Path(file_path)
    repo_root = Path(repo_root)

    ext = file_path.suffix
    language = _EXTENSION_MAP.get(ext)

    if language is None:
        raise ValueError(f"Unsupported file extension '{ext}' for: {file_path}")

    source_bytes = file_path.read_bytes()
    rel_path = str(file_path.relative_to(repo_root))

    parser = Parser(language)
    tree = parser.parse(source_bytes)

    result = ParsedFile(file_path=rel_path)

    if ext == ".py":
        _walk_tree(tree.root_node, result, source_bytes, rel_path)
    else:
        _walk_js_tree(tree.root_node, result, source_bytes, rel_path)

    return result


def parse_directory(repo_root: Path) -> list[ParsedFile]:
    """
    Walk repo_root recursively, parse all supported files, and return results.
    Skips files inside __pycache__, .git, .venv, node_modules, dist, build.

    Args:
        repo_root: Absolute path to the root of the project.

    Returns:
        List of ParsedFile, one per supported file found.
    """
    repo_root = Path(repo_root).resolve()
    results = []

    # Collect all files matching supported extensions
    all_files = []
    for ext in SUPPORTED_EXTENSIONS:
        all_files.extend(repo_root.rglob(f"*{ext}"))

    for source_file in sorted(set(all_files)):
        # Check if any parent directory is in SKIP_DIRS
        parts = source_file.relative_to(repo_root).parts
        if any(part in SKIP_DIRS for part in parts):
            continue

        try:
            parsed = parse_file(source_file, repo_root)
            results.append(parsed)
        except (ValueError, OSError) as e:
            logger.debug("Skipping %s: %s", source_file, e)
            continue

    return results
