# resolver.py
"""
Resolution phase: convert raw syntactic names from the parser into absolute FQNs.

Sits between parser.py (Phase 1) and ingest.py (Phase 3). Operates entirely
in-memory — no graph or file I/O.

Public API:
    build_module_fqn_map(parsed_files)                          → dict[file_path, module_fqn]
    build_symbol_table(pf, module_fqn, map)                     → SymbolTable
    build_reexport_map(parsed_files, module_fqn_map)            → dict[alias_fqn, canonical_fqn]
    canonicalize_fqn(fqn, reexport_map)                         → str
    enrich_star_imports(parsed_files, symbol_tables, fqn_map)   → None (mutates tables)
    resolve_callee(expr, caller_qualname, table, known_fqns)    → str | None
    resolve_base_class(name, table)                             → str | None
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from parser import ParsedFile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SymbolTable
# ---------------------------------------------------------------------------

@dataclass
class SymbolTable:
    module_fqn: str
    names: dict[str, str] = field(default_factory=dict)   # local_name → absolute_fqn
    has_star_import: bool = False                          # from X import * found

    def resolve(self, name: str) -> Optional[str]:
        return self.names.get(name)


# ---------------------------------------------------------------------------
# Module FQN map
# ---------------------------------------------------------------------------

def build_module_fqn_map(parsed_files: list["ParsedFile"]) -> dict[str, str]:
    """Return {file_path: module_fqn} for every parsed file.

    Examples:
        fastapi/routing.py      → fastapi.routing
        fastapi/__init__.py     → fastapi
        mypackage/sub/utils.py  → mypackage.sub.utils
    """
    result: dict[str, str] = {}
    for pf in parsed_files:
        result[pf.file_path] = _file_path_to_module_fqn(pf.file_path)
    return result


def _file_path_to_module_fqn(file_path: str) -> str:
    """Convert a relative file path to a dotted module FQN."""
    path = Path(file_path.replace("\\", "/"))
    if path.name == "__init__.py":
        parts = list(path.parent.parts)
    else:
        parts = list(path.with_suffix("").parts)
    return ".".join(p for p in parts if p and p != ".")


# ---------------------------------------------------------------------------
# Symbol table construction
# ---------------------------------------------------------------------------

def build_symbol_table(
    pf: "ParsedFile",
    module_fqn: str,
    module_fqn_map: dict[str, str],
) -> SymbolTable:
    """Build a {local_name: absolute_fqn} table for a single module.

    Sources (in order applied, last wins for collisions):
      1. Import statements (from X import Y, import X as Z)
      2. Own top-level class definitions
      3. Own top-level function definitions
    """
    table = SymbolTable(module_fqn=module_fqn)

    for imp in pf.imports:
        _process_import(imp, module_fqn, module_fqn_map, table)

    # Own top-level definitions — these shadow any imported name with the same spelling
    for cls in pf.classes:
        # Top-level only: qualname has no "." (e.g., "Foo" not "Outer.Foo")
        if "." not in cls.qualname:
            table.names[cls.name] = f"{module_fqn}.{cls.name}"

    for func in pf.functions:
        # Top-level only: qualname equals name (no enclosing class or function)
        if func.qualname == func.name:
            table.names[func.name] = f"{module_fqn}.{func.name}"

    # Fix E: resolve assignment aliases in __init__.py files.
    # e.g. `Model = _Model` where `_Model` was imported → `Model` gets same canonical FQN.
    # Must run after imports are processed so table.names is populated.
    if pf.file_path.endswith("__init__.py"):
        for alias_name, source_name in pf.top_level_aliases:
            if alias_name in table.names:
                continue  # already defined (own class/func wins)
            if source_name in table.names:
                table.names[alias_name] = table.names[source_name]
            else:
                # source_name is a local definition not yet in table (e.g., defined in this file)
                table.names[alias_name] = f"{module_fqn}.{source_name}"

    return table


def _process_import(imp, module_fqn: str, module_fqn_map: dict[str, str], table: SymbolTable) -> None:
    """Add a single ImportRef's bindings to the symbol table."""
    from parser import ImportRef  # local import to avoid circular at module load

    # Star import — we can't enumerate names, just flag it
    if imp.imported_name == "*":
        table.has_star_import = True
        return

    if imp.imported_name is not None:
        # from X import name [as alias]
        # Binds: alias (or name) → X.name
        local_name = imp.alias if imp.alias else imp.imported_name
        absolute_fqn = f"{imp.module}.{imp.imported_name}"
        table.names[local_name] = absolute_fqn
    else:
        # import X [as alias]
        # Binds: alias (or first component of X) → X
        if imp.alias:
            table.names[imp.alias] = imp.module
        else:
            # `import os.path` binds the name "os", not "os.path"
            top_level = imp.module.split(".")[0]
            table.names[top_level] = imp.module


# ---------------------------------------------------------------------------
# Re-export map  (Fix 1 — __init__.py re-exports)
# ---------------------------------------------------------------------------

def build_reexport_map(
    parsed_files: list["ParsedFile"],
    module_fqn_map: dict[str, str],
) -> dict[str, str]:
    """Scan __init__.py files and return alias_fqn → canonical_fqn.

    When django/db/models/__init__.py does `from .base import Model`, any
    caller that resolves `django.db.models.Model.save` actually needs the
    node at `django.db.models.base.Model.save`.  This map enables that fix.

    Example output:
        {"django.db.models.Model": "django.db.models.base.Model"}
    """
    reexport_map: dict[str, str] = {}
    for pf in parsed_files:
        if not pf.file_path.endswith("__init__.py"):
            continue
        pkg_module = module_fqn_map[pf.file_path]   # e.g. "django.db.models"
        for imp in pf.imports:
            if imp.imported_name is None or imp.imported_name == "*":
                continue
            alias_name    = imp.alias if imp.alias else imp.imported_name
            alias_fqn     = f"{pkg_module}.{alias_name}"
            canonical_fqn = f"{imp.module}.{imp.imported_name}"
            if alias_fqn != canonical_fqn:
                reexport_map[alias_fqn] = canonical_fqn
    return reexport_map


def canonicalize_fqn(fqn: str, reexport_map: dict[str, str]) -> str:
    """Replace any aliased prefix in *fqn* with its canonical form.

    Matches the longest prefix first; handles chained re-exports via
    one recursive call.

    Examples:
        "django.db.models.Model.save"     → "django.db.models.base.Model.save"
        "django.db.models.Model"          → "django.db.models.base.Model"
        "myapp.utils.helper"              → unchanged (no alias)
    """
    if not reexport_map:
        return fqn
    parts = fqn.split(".")
    for i in range(len(parts), 0, -1):
        prefix = ".".join(parts[:i])
        if prefix in reexport_map:
            canonical_prefix = reexport_map[prefix]
            suffix = ".".join(parts[i:])
            result = f"{canonical_prefix}.{suffix}" if suffix else canonical_prefix
            # One more pass handles chained re-exports (A → B → C)
            return canonicalize_fqn(result, reexport_map)
    return fqn


# ---------------------------------------------------------------------------
# Star import enrichment  (Fix 2 — within-repo star imports)
# ---------------------------------------------------------------------------

def enrich_star_imports(
    parsed_files: list["ParsedFile"],
    symbol_tables: dict[str, "SymbolTable"],
    module_fqn_map: dict[str, str],
) -> None:
    """Resolve within-repo star imports by enumerating source module exports.

    For `from mypackage.utils import *` where mypackage.utils is in our parsed
    file set, this adds all of utils' top-level names to the importing file's
    symbol table so they can be resolved to full FQNs.

    External package star imports (os, django.shortcuts, etc.) are left
    unresolved — the Jedi pass handles those.
    """
    # Build module_fqn → set of top-level exported names
    module_exports: dict[str, set[str]] = {}
    # Fix D: also record explicit __all__ restrictions per module
    module_all: dict[str, set[str]] = {}
    for pf in parsed_files:
        mod = module_fqn_map[pf.file_path]
        names: set[str] = set()
        for cls in pf.classes:
            if "." not in cls.qualname:          # top-level only
                names.add(cls.name)
        for func in pf.functions:
            if func.qualname == func.name:       # top-level only
                names.add(func.name)
        module_exports[mod] = names
        if pf.exports:
            module_all[mod] = set(pf.exports)

    for pf in parsed_files:
        table = symbol_tables[pf.file_path]
        if not table.has_star_import:
            continue
        for imp in pf.imports:
            if imp.imported_name != "*":
                continue
            if imp.module not in module_exports:
                continue   # external package — leave for Jedi
            names_to_add = module_exports[imp.module]
            # Fix D: respect __all__ when defined — only add names listed there
            if imp.module in module_all:
                names_to_add = names_to_add & module_all[imp.module]
            for name in names_to_add:
                if name not in table.names:      # don't shadow explicit imports
                    table.names[name] = f"{imp.module}.{name}"


# ---------------------------------------------------------------------------
# Callee resolution
# ---------------------------------------------------------------------------

def resolve_callee(
    callee_expr: str,
    caller_qualname: str,
    table: SymbolTable,
    known_fqns: set[str],
) -> Optional[str]:
    """Resolve a raw callee expression to an absolute FQN.

    Returns None if unresolvable — the caller should pass this to the Jedi pass.

    Resolution strategy:
      1. self / cls  → current class in same module
      2. head in symbol table → substitute and append tail
      3. head is a known local FQN → use directly
      4. → None (Jedi)

    Args:
        callee_expr:     Full expression from parser, e.g. "self.connect", "os.path.join", "helper"
        caller_qualname: Scope of the call site, e.g. "DBManager.connect" or "MyClass.__classbody__"
        table:           Symbol table for the module containing the call
        known_fqns:      Set of all Function/Class FQNs written to the graph so far
    """
    if not callee_expr:
        return None

    parts = callee_expr.split(".", maxsplit=1)
    head = parts[0]
    tail = parts[1] if len(parts) > 1 else None

    # --- Case 1: self / cls / super().method ---
    if head in ("self", "cls") or head.startswith("super("):
        if not tail:
            return None  # bare `self` or `cls` is not a call target
        class_name = _extract_class_from_qualname(caller_qualname)
        if not class_name:
            return None
        candidate = f"{table.module_fqn}.{class_name}.{tail}"
        # Accept even if not in known_fqns — may be inherited; Jedi will upgrade if wrong
        return candidate

    # --- Case 2: head found in symbol table ---
    resolved_head = table.resolve(head)
    if resolved_head is not None:
        if tail:
            return f"{resolved_head}.{tail}"
        return resolved_head

    # --- Case 3: head is a locally-defined name (not yet in symbol table entries) ---
    local_candidate = f"{table.module_fqn}.{head}"
    if local_candidate in known_fqns:
        if tail:
            return f"{local_candidate}.{tail}"
        return local_candidate

    # --- Unresolvable → Jedi ---
    return None


def _extract_class_from_qualname(qualname: str) -> Optional[str]:
    """Extract the class name from a method's qualname.

    Examples:
        "MyClass.my_method"       → "MyClass"
        "Outer.Inner.method"      → "Outer.Inner"
        "standalone_function"     → None
        "<module>"                → None
        "MyClass.__classbody__"   → "MyClass"
    """
    if not qualname or qualname in ("<module>", "<locals>"):
        return None
    parts = qualname.rsplit(".", maxsplit=1)
    if len(parts) < 2:
        return None  # top-level function, no enclosing class
    class_part = parts[0]
    # Filter out <locals> sentinel — nested function inside module-level function
    if class_part == "<locals>":
        return None
    return class_part


# ---------------------------------------------------------------------------
# Base class resolution
# ---------------------------------------------------------------------------

def resolve_base_class(base_name: str, table: SymbolTable) -> Optional[str]:
    """Resolve a raw base class name to an absolute FQN.

    Examples (given appropriate symbol table):
        "Animal"    → "zoo.Animal"         (if Animal imported from zoo)
        "BaseModel" → "pydantic.BaseModel" (if BaseModel imported from pydantic)
        "dict"      → "builtins.dict"      (builtin)
        "Unknown"   → None                 (not in table, not a builtin)

    Returns None if unresolvable — callee stored with unresolved flag.
    """
    if not base_name:
        return None

    # Handle dotted base names like `module.ClassName`
    parts = base_name.split(".", maxsplit=1)
    head = parts[0]
    tail = parts[1] if len(parts) > 1 else None

    resolved = table.resolve(head)
    if resolved is not None:
        return f"{resolved}.{tail}" if tail else resolved

    # Check builtins
    if base_name in _PYTHON_BUILTINS:
        return f"builtins.{base_name}"

    # Not resolvable
    return None


# Common Python builtins that appear as base classes
_PYTHON_BUILTINS = frozenset({
    "object", "int", "float", "str", "bytes", "bool", "list", "dict",
    "tuple", "set", "frozenset", "type", "Exception", "BaseException",
    "ValueError", "TypeError", "KeyError", "AttributeError", "RuntimeError",
    "NotImplementedError", "StopIteration", "GeneratorExit", "OSError",
    "IOError", "FileNotFoundError", "PermissionError", "IndexError",
    "OverflowError", "ZeroDivisionError", "MemoryError", "RecursionError",
    "ImportError", "ModuleNotFoundError", "NameError", "UnboundLocalError",
    "AssertionError", "ArithmeticError", "LookupError", "SyntaxError",
    "UnicodeError", "UnicodeDecodeError", "UnicodeEncodeError",
    "enum.Enum", "enum.IntEnum", "enum.Flag",
})
