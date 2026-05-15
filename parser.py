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


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FunctionDef:
    name: str
    file_path: str
    start_line: int
    end_line: int
    docstring: Optional[str]
    is_method: bool
    class_name: Optional[str]        # immediate enclosing class (for schema compat)
    qualname: str = ""               # Python __qualname__ style: "Outer.Inner.method"
    params: list[str] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)
    return_annotation: Optional[str] = None
    raises: list[str] = field(default_factory=list)   # exception type names from raise stmts


@dataclass
class ClassDef:
    name: str
    file_path: str
    start_line: int
    end_line: int
    docstring: Optional[str]
    qualname: str = ""               # "OuterClass.InnerClass" for nested
    bases: list[str] = field(default_factory=list)


@dataclass
class ImportRef:
    file_path: str
    module: str                      # absolute module path after relative resolution
    alias: Optional[str]             # as-alias (for the imported name or module)
    imported_name: Optional[str] = None  # name FROM module; None for plain `import X`


@dataclass
class CallEdge:
    caller_qualname: str             # scope stack qualname, e.g. "ClassName.method" or "<module>"
    callee_expr: str                 # full expression: "self.connect", "os.path.join", "helper"
    file_path: str
    line: int
    column: int = 0                  # column of the METHOD NAME (not start of call expression)


@dataclass
class VariableRef:
    """A non-call reference to an identifier inside a function body.

    Used to detect cross-module variable reads (e.g. `from config import DB_URL`
    then `engine.connect(DB_URL)` — DB_URL is read but not called).
    Filtered against the symbol table in ingest.py; only cross-module reads
    survive to become READS edges.
    """
    user_qualname: str   # scope: "ClassName.method", "func_name", or "<module>"
    name: str            # the identifier text, e.g. "DATABASE_URL"
    file_path: str
    line: int


@dataclass
class ParsedFile:
    file_path: str
    functions: list[FunctionDef] = field(default_factory=list)
    classes: list[ClassDef] = field(default_factory=list)
    imports: list[ImportRef] = field(default_factory=list)
    calls: list[CallEdge] = field(default_factory=list)
    variable_refs: list[VariableRef] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)
    # Fix D: names listed in __all__ (empty = no __all__ defined = all names exported)
    top_level_aliases: list[tuple[str, str]] = field(default_factory=list)
    # Fix E: module-level Name = OtherName assignments (alias_name, source_name)


# ---------------------------------------------------------------------------
# Python AST helpers
# ---------------------------------------------------------------------------

def _extract_docstring(node) -> Optional[str]:
    body = node.child_by_field_name("body")
    if body is None:
        return None
    for child in body.children:
        if child.type in ("newline", "indent", "dedent", "comment"):
            continue
        if child.type == "expression_statement":
            string_node = child.children[0] if child.children else None
            if string_node and string_node.type == "string":
                raw = string_node.text.decode("utf-8")
                for delim in ('"""', "'''"):
                    if raw.startswith(delim) and raw.endswith(delim):
                        raw = raw[3:-3]
                        break
                else:
                    if raw.startswith('"') and raw.endswith('"'):
                        raw = raw[1:-1]
                    elif raw.startswith("'") and raw.endswith("'"):
                        raw = raw[1:-1]
                return raw.strip()
            break
        else:
            break
    return None


def _resolve_callee_expr(call_node) -> Optional[tuple[str, int]]:
    """Return (full_callee_expression, column_of_method_name) or None.

    For attribute calls (obj.method), column points to 'method', not 'obj',
    because jedi.goto() at the method name resolves the definition correctly.

    Examples:
        foo()           → ("foo",       col_of_f)
        self.connect()  → ("self.connect", col_of_connect)
        os.path.join()  → ("os.path.join", col_of_join)
        obj.run()       → ("obj.run",   col_of_run)
    """
    func = call_node.child_by_field_name("function")
    if func is None:
        return None

    if func.type == "identifier":
        return func.text.decode("utf-8"), func.start_point[1]

    elif func.type == "attribute":
        obj = func.child_by_field_name("object")
        attr = func.child_by_field_name("attribute")
        if obj and attr:
            obj_text = obj.text.decode("utf-8")
            attr_text = attr.text.decode("utf-8")
            # Column points to the attribute (method name), not the object
            return f"{obj_text}.{attr_text}", attr.start_point[1]

    return None


def _extract_import_refs(node, rel_path: str) -> list[ImportRef]:
    """Extract ImportRef objects from import_statement or import_from_statement.

    Emits one ImportRef per bound name so the resolver can build exact mappings:
      from os.path import join, exists  → [ImportRef(os.path, None, join),
                                           ImportRef(os.path, None, exists)]
      import numpy as np                → [ImportRef(numpy, np, None)]
      from . import utils               → [ImportRef(pkg.utils, None, None)] (after resolution)
    """
    refs = []

    if node.type == "import_statement":
        for child in node.children:
            if child.type == "dotted_name":
                module = child.text.decode("utf-8")
                module = _resolve_relative_import(module, rel_path)
                refs.append(ImportRef(file_path=rel_path, module=module, alias=None, imported_name=None))
            elif child.type == "aliased_import":
                name_node = child.child_by_field_name("name")
                alias_node = child.child_by_field_name("alias")
                if name_node:
                    module = name_node.text.decode("utf-8")
                    module = _resolve_relative_import(module, rel_path)
                    alias = alias_node.text.decode("utf-8") if alias_node else None
                    refs.append(ImportRef(file_path=rel_path, module=module, alias=alias, imported_name=None))

    elif node.type == "import_from_statement":
        module_node = node.child_by_field_name("module_name")
        if module_node:
            raw_module = module_node.text.decode("utf-8")
        else:
            raw_module = None
            for child in node.children:
                if child.type in ("dotted_name", "relative_import"):
                    raw_module = child.text.decode("utf-8")
                    break
        if raw_module is None:
            return refs

        module = _resolve_relative_import(raw_module, rel_path)

        # Collect all imported names (handles multiple names and aliases)
        for child in node.children:
            if child.type == "wildcard_import":
                # from X import *
                refs.append(ImportRef(file_path=rel_path, module=module, alias=None, imported_name="*"))
                return refs

            elif child.type == "dotted_name" and child != module_node:
                # from X import Name  (plain name)
                imported_name = child.text.decode("utf-8")
                refs.append(ImportRef(file_path=rel_path, module=module, alias=None, imported_name=imported_name))

            elif child.type == "aliased_import":
                # from X import Name as Alias
                name_node = child.child_by_field_name("name")
                alias_node = child.child_by_field_name("alias")
                if name_node:
                    imported_name = name_node.text.decode("utf-8")
                    alias = alias_node.text.decode("utf-8") if alias_node else None
                    refs.append(ImportRef(file_path=rel_path, module=module, alias=alias, imported_name=imported_name))

    return refs


def _resolve_relative_import(module: str, source_file_path: str) -> str:
    if not module.startswith("."):
        return module
    level = len(module) - len(module.lstrip("."))
    rel_part = module.lstrip(".")
    parts = source_file_path.replace("\\", "/").replace(".py", "").split("/")
    base_parts = parts[:-level] if level <= len(parts) else []
    base = ".".join(p for p in base_parts if p and p != ".")
    if rel_part:
        return f"{base}.{rel_part}" if base else rel_part
    return base if base else "."


# ---------------------------------------------------------------------------
# Python AST walker — scope-stack based
# ---------------------------------------------------------------------------

def _walk_tree(
    node,
    result: ParsedFile,
    source_bytes: bytes,
    rel_path: str,
    class_stack: list[str],    # names of enclosing classes (innermost last)
    func_stack: list[str],     # names of enclosing functions (innermost last)
) -> None:
    """Recursively walk the Python AST, tracking full scope via explicit stacks."""

    if node.type == "class_definition":
        name_node = node.child_by_field_name("name")
        if name_node:
            class_name = name_node.text.decode("utf-8")
            qualname = ".".join(class_stack + [class_name])

            # Base classes
            bases = []
            superclasses_node = node.child_by_field_name("superclasses")
            if superclasses_node:
                for child in superclasses_node.children:
                    if child.type in ("identifier", "attribute"):
                        bases.append(child.text.decode("utf-8"))

            result.classes.append(ClassDef(
                name=class_name,
                file_path=rel_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                docstring=_extract_docstring(node),
                qualname=qualname,
                bases=bases,
            ))

            # Recurse into class body with updated class stack
            for child in node.children:
                _walk_tree(child, result, source_bytes, rel_path,
                           class_stack + [class_name], func_stack)
        return  # already recursed

    elif node.type == "function_definition":
        name_node = node.child_by_field_name("name")
        if name_node:
            func_name = name_node.text.decode("utf-8")

            # qualname: use <locals> whenever inside another function (Python __qualname__ convention)
            if func_stack:
                # Nested inside another function — class context (if any) + enclosing func + <locals>
                if class_stack:
                    qualname = ".".join(class_stack + [func_stack[-1], "<locals>", func_name])
                else:
                    qualname = ".".join(func_stack + ["<locals>", func_name])
            else:
                qualname = ".".join(class_stack + [func_name])

            immediate_class = class_stack[-1] if class_stack else None

            # Parameters
            params = []
            params_node = node.child_by_field_name("parameters")
            if params_node:
                for child in params_node.children:
                    if child.type == "identifier":
                        params.append(child.text.decode("utf-8"))
                    elif child.type in ("typed_parameter", "default_parameter",
                                        "typed_default_parameter"):
                        pname = child.child_by_field_name("name")
                        if pname is None and child.children:
                            pname = child.children[0]
                        if pname:
                            params.append(pname.text.decode("utf-8"))

            # Decorators — walk backwards from prev siblings
            decorators = []
            sibling = node.prev_sibling
            while sibling:
                if sibling.type == "decorator":
                    decorators.append(sibling.text.decode("utf-8").lstrip("@").strip())
                    sibling = sibling.prev_sibling
                elif sibling.type in ("newline", "comment"):
                    sibling = sibling.prev_sibling
                else:
                    break

            # Return annotation
            return_annotation = None
            return_type_node = node.child_by_field_name("return_type")
            if return_type_node:
                return_annotation = return_type_node.text.decode("utf-8").lstrip("->").strip()

            result.functions.append(FunctionDef(
                name=func_name,
                file_path=rel_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                docstring=_extract_docstring(node),
                is_method=immediate_class is not None,
                class_name=immediate_class,
                qualname=qualname,
                params=params,
                decorators=decorators,
                return_annotation=return_annotation,
            ))

            # Recurse — push function onto func_stack (for <locals> nesting)
            for child in node.children:
                _walk_tree(child, result, source_bytes, rel_path,
                           class_stack, func_stack + [func_name])
        return  # already recursed

    elif node.type in ("import_statement", "import_from_statement"):
        for ref in _extract_import_refs(node, rel_path):
            result.imports.append(ref)

    elif node.type == "call":
        resolved = _resolve_callee_expr(node)
        if resolved:
            callee_expr, col = resolved
            # Determine caller context from stacks
            if func_stack:
                if class_stack:
                    caller_qualname = ".".join(class_stack + [func_stack[-1]])
                else:
                    # Inside a nested function — use innermost function
                    caller_qualname = func_stack[-1]
            elif class_stack:
                caller_qualname = ".".join(class_stack) + ".__classbody__"
            else:
                caller_qualname = "<module>"

            result.calls.append(CallEdge(
                caller_qualname=caller_qualname,
                callee_expr=callee_expr,
                file_path=rel_path,
                line=node.start_point[0] + 1,
                column=col,
            ))

    elif node.type == "attribute" and func_stack:
        # Fix C: collect compound attribute reads like `config.DATABASE_URL`.
        # Only simple obj.attr where obj is a plain identifier — not calls, not
        # nested attributes like os.path.join (outer node has attribute as object).
        obj  = node.child_by_field_name("object")
        attr = node.child_by_field_name("attribute")
        if obj and attr and obj.type == "identifier":
            parent = node.parent
            is_callee = (
                parent and parent.type == "call"
                and (fn := parent.child_by_field_name("function")) is not None
                and fn.start_point == node.start_point and fn.end_point == node.end_point
            )
            if not is_callee:
                if class_stack:
                    user_qualname = ".".join(class_stack + [func_stack[-1]])
                else:
                    user_qualname = func_stack[-1]
                compound = f"{obj.text.decode('utf-8')}.{attr.text.decode('utf-8')}"
                result.variable_refs.append(VariableRef(
                    user_qualname=user_qualname,
                    name=compound,
                    file_path=rel_path,
                    line=node.start_point[0] + 1,
                ))

    elif node.type == "assignment" and not func_stack and not class_stack:
        left  = node.child_by_field_name("left")
        right = node.child_by_field_name("right")
        if left and left.type == "identifier" and right:
            left_name = left.text.decode("utf-8")
            # Fix D: __all__ = ["Foo", "Bar"] — record exported names
            if left_name == "__all__" and right.type in ("list", "tuple"):
                for child in right.named_children:
                    if child.type == "string":
                        val = child.text.decode("utf-8").strip("'\" \n")
                        if val:
                            result.exports.append(val)
            # Fix E: Model = _Model — simple name alias at module level
            elif left_name != "__all__" and right.type == "identifier":
                result.top_level_aliases.append(
                    (left_name, right.text.decode("utf-8"))
                )

    elif node.type == "identifier" and func_stack:
        # Collect non-call identifier reads for READS edge tracking.
        # Only collected inside function bodies (func_stack non-empty).
        # Filtered against the symbol table in ingest.py — only cross-module
        # imported names that are actually read survive to become READS edges.
        name = node.text.decode("utf-8")
        parent = node.parent

        skip = (
            len(name) < 2
            # Part of dotted attribute access obj.attr — parent handles it
            or (parent and parent.type == "attribute")
            # This identifier IS the callee of a call — already a CALLS edge
            or (parent and parent.type == "call"
                and (fn := parent.child_by_field_name("function")) is not None
                and fn.start_point == node.start_point and fn.end_point == node.end_point)
            # Definition contexts — this is a name being defined, not read
            or (parent and parent.type in (
                "function_definition", "class_definition",
                "parameters", "typed_parameter", "default_parameter",
                "typed_default_parameter", "lambda_parameters",
                "import_statement", "import_from_statement",
                "aliased_import", "as_pattern",
                "for_statement",   # loop variable
            ))
        )
        if not skip:
            if func_stack:
                if class_stack:
                    user_qualname = ".".join(class_stack + [func_stack[-1]])
                else:
                    user_qualname = func_stack[-1]
            else:
                user_qualname = "<module>"
            result.variable_refs.append(VariableRef(
                user_qualname=user_qualname,
                name=name,
                file_path=rel_path,
                line=node.start_point[0] + 1,
            ))

    elif node.type == "raise_statement" and func_stack and result.functions:
        # Extract exception type name from `raise ExcType(...)` or `raise ExcType`
        # for Tier 0 fingerprint raises field.
        for child in node.named_children:
            if child.type in ("call", "identifier", "attribute"):
                # Get the base name: ExcType(...) → ExcType, mod.ExcType → ExcType
                base = child.child_by_field_name("function") or child
                exc_text = base.text.decode("utf-8").split("(")[0].split(".")[-1]
                if exc_text and exc_text[0].isupper():
                    # Attach to the innermost function definition
                    for func in reversed(result.functions):
                        # Only attach if this raise is inside the function's line range
                        raise_line = node.start_point[0] + 1
                        if func.start_line <= raise_line <= func.end_line:
                            if exc_text not in func.raises:
                                func.raises.append(exc_text)
                            break
                break

    # Recurse into children for all other node types
    for child in node.children:
        _walk_tree(child, result, source_bytes, rel_path, class_stack, func_stack)


# ---------------------------------------------------------------------------
# JavaScript / TypeScript AST walking
# ---------------------------------------------------------------------------

def _extract_js_docstring(node) -> Optional[str]:
    prev = node.prev_sibling
    if prev and prev.type == "comment":
        text = prev.text.decode("utf-8").strip()
        if text.startswith("/**") and text.endswith("*/"):
            text = text[3:-2].strip()
        elif text.startswith("//"):
            text = text[2:].strip()
        return text
    return None


def _walk_js_tree(
    node,
    result: ParsedFile,
    source_bytes: bytes,
    rel_path: str,
    class_stack: list[str],
    func_stack: list[str],
) -> None:
    if node.type in ("class_declaration", "class"):
        name_node = node.child_by_field_name("name")
        if name_node:
            class_name = name_node.text.decode("utf-8")
            qualname = ".".join(class_stack + [class_name])

            bases = []
            heritage = node.child_by_field_name("heritage")
            if heritage is None:
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
                docstring=_extract_js_docstring(node),
                qualname=qualname,
                bases=bases,
            ))
            body = node.child_by_field_name("body")
            if body:
                for child in body.children:
                    _walk_js_tree(child, result, source_bytes, rel_path,
                                  class_stack + [class_name], func_stack)
        return

    elif node.type in ("function_declaration", "method_definition",
                       "arrow_function", "function"):
        name_node = node.child_by_field_name("name")
        func_name = None
        if name_node:
            func_name = name_node.text.decode("utf-8")
        elif node.type == "arrow_function" and node.parent:
            if node.parent.type == "variable_declarator":
                var_name = node.parent.child_by_field_name("name")
                if var_name:
                    func_name = var_name.text.decode("utf-8")

        if func_name:
            qualname = ".".join(class_stack + [func_name]) if class_stack else func_name
            is_method = bool(class_stack) or node.type == "method_definition"
            result.functions.append(FunctionDef(
                name=func_name,
                file_path=rel_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                docstring=_extract_js_docstring(node),
                is_method=is_method,
                class_name=class_stack[-1] if class_stack else None,
                qualname=qualname,
            ))
            for child in node.children:
                _walk_js_tree(child, result, source_bytes, rel_path,
                              class_stack, func_stack + [func_name])
        return

    elif node.type in ("import_statement", "import_declaration"):
        source_node = node.child_by_field_name("source")
        if source_node:
            module = source_node.text.decode("utf-8").strip("'\"")
            module = _resolve_relative_import(module, rel_path)
            result.imports.append(ImportRef(
                file_path=rel_path, module=module, alias=None, imported_name=None,
            ))

    elif node.type == "call_expression":
        func = node.child_by_field_name("function")
        callee_expr = None
        col = node.start_point[1]
        if func:
            if func.type == "identifier":
                callee_expr = func.text.decode("utf-8")
                col = func.start_point[1]
            elif func.type == "member_expression":
                obj_node = func.child_by_field_name("object")
                prop_node = func.child_by_field_name("property")
                if obj_node and prop_node:
                    callee_expr = f"{obj_node.text.decode('utf-8')}.{prop_node.text.decode('utf-8')}"
                    col = prop_node.start_point[1]

        if callee_expr:
            if func_stack:
                caller_qualname = ".".join(class_stack + [func_stack[-1]]) if class_stack else func_stack[-1]
            elif class_stack:
                caller_qualname = ".".join(class_stack) + ".__classbody__"
            else:
                caller_qualname = "<module>"

            result.calls.append(CallEdge(
                caller_qualname=caller_qualname,
                callee_expr=callee_expr,
                file_path=rel_path,
                line=node.start_point[0] + 1,
                column=col,
            ))

    for child in node.children:
        _walk_js_tree(child, result, source_bytes, rel_path, class_stack, func_stack)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_file(file_path: Path, repo_root: Path) -> ParsedFile:
    """Parse a single source file and return all extracted entities.

    Supports: .py, .js, .jsx, .ts, .tsx
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
        _walk_tree(tree.root_node, result, source_bytes, rel_path,
                   class_stack=[], func_stack=[])
    else:
        _walk_js_tree(tree.root_node, result, source_bytes, rel_path,
                      class_stack=[], func_stack=[])

    return result


def parse_directory(repo_root: Path) -> list[ParsedFile]:
    """Walk repo_root recursively, parse all supported files."""
    repo_root = Path(repo_root).resolve()
    results = []

    all_files = []
    for ext in SUPPORTED_EXTENSIONS:
        all_files.extend(repo_root.rglob(f"*{ext}"))

    for source_file in sorted(set(all_files)):
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
