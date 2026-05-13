"""Unit tests for resolver.py — no FalkorDB required."""

import pytest
from pathlib import Path

from parser import parse_file, ParsedFile, ImportRef
from resolver import (
    SymbolTable,
    build_module_fqn_map,
    build_symbol_table,
    build_reexport_map,
    canonicalize_fqn,
    enrich_star_imports,
    resolve_callee,
    resolve_base_class,
    _file_path_to_module_fqn,
)

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# _file_path_to_module_fqn
# ---------------------------------------------------------------------------

class TestFilePathToModuleFqn:
    def test_simple(self):
        assert _file_path_to_module_fqn("fastapi/routing.py") == "fastapi.routing"

    def test_init(self):
        assert _file_path_to_module_fqn("fastapi/__init__.py") == "fastapi"

    def test_nested(self):
        assert _file_path_to_module_fqn("mypackage/sub/utils.py") == "mypackage.sub.utils"

    def test_top_level(self):
        assert _file_path_to_module_fqn("ingest.py") == "ingest"


# ---------------------------------------------------------------------------
# build_module_fqn_map
# ---------------------------------------------------------------------------

class TestBuildModuleFqnMap:
    def test_maps_fixture_files(self):
        pf_simple = parse_file(FIXTURES / "simple_module.py", FIXTURES)
        pf_caller = parse_file(FIXTURES / "caller_module.py", FIXTURES)
        fqn_map = build_module_fqn_map([pf_simple, pf_caller])
        assert fqn_map["simple_module.py"] == "simple_module"
        assert fqn_map["caller_module.py"] == "caller_module"


# ---------------------------------------------------------------------------
# build_symbol_table — import rules
# ---------------------------------------------------------------------------

class TestBuildSymbolTable:
    def _make_pf(self, file_path: str, imports: list[ImportRef]) -> ParsedFile:
        pf = ParsedFile(file_path=file_path)
        pf.imports = imports
        return pf

    def test_plain_import(self):
        pf = self._make_pf("mymod.py", [ImportRef("mymod.py", "os", None, None)])
        table = build_symbol_table(pf, "mymod", {"mymod.py": "mymod"})
        assert table.resolve("os") == "os"

    def test_plain_import_dotted_binds_top(self):
        # `import os.path` → binds "os" → "os.path"
        pf = self._make_pf("mymod.py", [ImportRef("mymod.py", "os.path", None, None)])
        table = build_symbol_table(pf, "mymod", {"mymod.py": "mymod"})
        assert table.resolve("os") == "os.path"

    def test_import_alias(self):
        # import numpy as np
        pf = self._make_pf("mymod.py", [ImportRef("mymod.py", "numpy", "np", None)])
        table = build_symbol_table(pf, "mymod", {"mymod.py": "mymod"})
        assert table.resolve("np") == "numpy"
        assert table.resolve("numpy") is None

    def test_from_import(self):
        # from os.path import join
        pf = self._make_pf("mymod.py", [ImportRef("mymod.py", "os.path", None, "join")])
        table = build_symbol_table(pf, "mymod", {"mymod.py": "mymod"})
        assert table.resolve("join") == "os.path.join"

    def test_from_import_alias(self):
        # from os.path import join as j
        pf = self._make_pf("mymod.py", [ImportRef("mymod.py", "os.path", "j", "join")])
        table = build_symbol_table(pf, "mymod", {"mymod.py": "mymod"})
        assert table.resolve("j") == "os.path.join"
        assert table.resolve("join") is None

    def test_star_import_sets_flag(self):
        pf = self._make_pf("mymod.py", [ImportRef("mymod.py", "somemod", None, "*")])
        table = build_symbol_table(pf, "mymod", {"mymod.py": "mymod"})
        assert table.has_star_import is True

    def test_own_class_added(self):
        from parser import ClassDef
        pf = self._make_pf("zoo.py", [])
        cls = ClassDef(name="Animal", file_path="zoo.py", start_line=1, end_line=5,
                       docstring=None, qualname="Animal")
        pf.classes = [cls]
        table = build_symbol_table(pf, "zoo", {"zoo.py": "zoo"})
        assert table.resolve("Animal") == "zoo.Animal"

    def test_own_function_added(self):
        from parser import FunctionDef
        pf = self._make_pf("zoo.py", [])
        fn = FunctionDef(name="helper", file_path="zoo.py", start_line=1, end_line=3,
                         docstring=None, is_method=False, class_name=None, qualname="helper")
        pf.functions = [fn]
        table = build_symbol_table(pf, "zoo", {"zoo.py": "zoo"})
        assert table.resolve("helper") == "zoo.helper"

    def test_nested_class_not_added_to_top_level(self):
        from parser import ClassDef
        pf = self._make_pf("zoo.py", [])
        inner = ClassDef(name="Inner", file_path="zoo.py", start_line=3, end_line=5,
                         docstring=None, qualname="Outer.Inner")
        pf.classes = [inner]
        table = build_symbol_table(pf, "zoo", {"zoo.py": "zoo"})
        # Nested class is not a top-level name
        assert table.resolve("Inner") is None

    def test_resolution_caller_fixture(self):
        pf = parse_file(FIXTURES / "resolution_caller.py", FIXTURES)
        fqn_map = build_module_fqn_map([pf])
        table = build_symbol_table(pf, fqn_map[pf.file_path], fqn_map)
        # from tests.fixtures.resolution_target import Animal, standalone_helper
        assert table.resolve("Animal") == "tests.fixtures.resolution_target.Animal"
        assert table.resolve("standalone_helper") == "tests.fixtures.resolution_target.standalone_helper"
        # import os.path as osp
        assert table.resolve("osp") == "os.path"


# ---------------------------------------------------------------------------
# resolve_callee
# ---------------------------------------------------------------------------

class TestResolveCallee:
    def _table(self, module_fqn: str, names: dict) -> SymbolTable:
        return SymbolTable(module_fqn=module_fqn, names=names)

    def test_simple_local(self):
        known = {"mymod.helper"}
        table = self._table("mymod", {})
        assert resolve_callee("helper", "mymod.process", table, known) == "mymod.helper"

    def test_imported_name(self):
        known = set()
        table = self._table("mymod", {"join": "os.path.join"})
        assert resolve_callee("join", "mymod.fn", table, known) == "os.path.join"

    def test_imported_module_dotted(self):
        # np.array where np → numpy
        known = set()
        table = self._table("mymod", {"np": "numpy"})
        assert resolve_callee("np.array", "mymod.fn", table, known) == "numpy.array"

    def test_self_method(self):
        known = {"mymod.DB.connect"}
        table = self._table("mymod", {})
        result = resolve_callee("self.connect", "DB.some_method", table, known)
        assert result == "mymod.DB.connect"

    def test_cls_method(self):
        known = {"mymod.Repo.create"}
        table = self._table("mymod", {})
        result = resolve_callee("cls.create", "Repo.from_path", table, known)
        assert result == "mymod.Repo.create"

    def test_self_no_class_context(self):
        # caller_qualname is a plain function (no enclosing class)
        table = self._table("mymod", {})
        result = resolve_callee("self.connect", "standalone_fn", table, set())
        assert result is None

    def test_unresolvable_local_var(self):
        # obj.method() where obj is a local variable — not in symbol table
        table = self._table("mymod", {})
        result = resolve_callee("obj.method", "mymod.fn", table, set())
        assert result is None

    def test_empty_expr(self):
        table = self._table("mymod", {})
        assert resolve_callee("", "mymod.fn", table, set()) is None

    def test_nested_class_self(self):
        # Method in Outer.Inner — caller_qualname = "Outer.Inner.method"
        known = {"mymod.Outer.Inner.helper"}
        table = self._table("mymod", {})
        result = resolve_callee("self.helper", "Outer.Inner.method", table, known)
        assert result == "mymod.Outer.Inner.helper"


# ---------------------------------------------------------------------------
# resolve_base_class
# ---------------------------------------------------------------------------

class TestResolveBaseClass:
    def _table(self, module_fqn: str, names: dict) -> SymbolTable:
        return SymbolTable(module_fqn=module_fqn, names=names)

    def test_imported_base(self):
        table = self._table("zoo", {"Animal": "biology.Animal"})
        assert resolve_base_class("Animal", table) == "biology.Animal"

    def test_same_module_base(self):
        table = self._table("zoo", {"Animal": "zoo.Animal"})
        assert resolve_base_class("Animal", table) == "zoo.Animal"

    def test_builtin_base(self):
        table = self._table("mymod", {})
        assert resolve_base_class("Exception", table) == "builtins.Exception"

    def test_dict_base(self):
        table = self._table("mymod", {})
        assert resolve_base_class("dict", table) == "builtins.dict"

    def test_unresolvable(self):
        table = self._table("mymod", {})
        assert resolve_base_class("UnknownBase", table) is None

    def test_pydantic_base_model(self):
        table = self._table("models", {"BaseModel": "pydantic.BaseModel"})
        assert resolve_base_class("BaseModel", table) == "pydantic.BaseModel"

    def test_super_resolves_like_self(self):
        # Fix B: super().method() should resolve to ChildClass.method
        # (Fix A's inheritance resolver then upgrades to Parent.method)
        table = SymbolTable(module_fqn="mymod")
        table.names["Animal"] = "zoo.Animal"
        known = {"mymod.Dog.speak", "zoo.Animal.speak"}
        result = resolve_callee("super().speak", "Dog.make_sound", table, known)
        assert result == "mymod.Dog.speak"

    def test_super_with_args_resolves(self):
        # super(Dog, self).speak() — obj is "super(Dog, self)" starts with "super("
        table = SymbolTable(module_fqn="mymod")
        known: set[str] = set()
        result = resolve_callee("super(Dog, self).speak", "Dog.make_sound", table, known)
        assert result == "mymod.Dog.speak"

    def test_resolution_caller_cat_inherits(self):
        # Cat(Animal) — Animal is imported from resolution_target
        pf = parse_file(FIXTURES / "resolution_caller.py", FIXTURES)
        fqn_map = build_module_fqn_map([pf])
        table = build_symbol_table(pf, fqn_map[pf.file_path], fqn_map)
        result = resolve_base_class("Animal", table)
        assert result == "tests.fixtures.resolution_target.Animal"


# ---------------------------------------------------------------------------
# build_reexport_map  (Fix 1)
# ---------------------------------------------------------------------------

class TestBuildReexportMap:
    def _make_pf(self, file_path: str, imports: list) -> "ParsedFile":
        from parser import ParsedFile
        return ParsedFile(file_path=file_path, functions=[], classes=[],
                          imports=imports, calls=[], variable_refs=[])

    def test_init_reexport_detected(self):
        from parser import ImportRef
        init_pf = self._make_pf(
            "pkg/__init__.py",
            [ImportRef(file_path="pkg/__init__.py", module="pkg.animals",
                       alias=None, imported_name="Animal")],
        )
        fqn_map = {"pkg/__init__.py": "pkg"}
        rmap = build_reexport_map([init_pf], fqn_map)
        assert rmap == {"pkg.Animal": "pkg.animals.Animal"}

    def test_non_init_file_ignored(self):
        from parser import ImportRef
        pf = self._make_pf(
            "pkg/utils.py",
            [ImportRef(file_path="pkg/utils.py", module="pkg.animals",
                       alias=None, imported_name="Animal")],
        )
        fqn_map = {"pkg/utils.py": "pkg.utils"}
        rmap = build_reexport_map([pf], fqn_map)
        assert rmap == {}

    def test_alias_used_in_reexport(self):
        from parser import ImportRef
        init_pf = self._make_pf(
            "myapp/__init__.py",
            [ImportRef(file_path="myapp/__init__.py", module="myapp.core",
                       alias="CoreModel", imported_name="Model")],
        )
        fqn_map = {"myapp/__init__.py": "myapp"}
        rmap = build_reexport_map([init_pf], fqn_map)
        assert rmap == {"myapp.CoreModel": "myapp.core.Model"}

    def test_star_import_in_init_skipped(self):
        from parser import ImportRef
        init_pf = self._make_pf(
            "pkg/__init__.py",
            [ImportRef(file_path="pkg/__init__.py", module="pkg.animals",
                       alias=None, imported_name="*")],
        )
        fqn_map = {"pkg/__init__.py": "pkg"}
        rmap = build_reexport_map([init_pf], fqn_map)
        assert rmap == {}

    def test_real_pkg_fixture(self):
        # Parse from project root so FQNs match absolute imports (tests.fixtures.pkg.*)
        PROJECT_ROOT = FIXTURES.parent.parent
        pf_init = parse_file(FIXTURES / "pkg" / "__init__.py", PROJECT_ROOT)
        pf_animals = parse_file(FIXTURES / "pkg" / "animals.py", PROJECT_ROOT)
        all_pfs = [pf_init, pf_animals]
        fqn_map = build_module_fqn_map(all_pfs)
        rmap = build_reexport_map(all_pfs, fqn_map)
        assert "tests.fixtures.pkg.Animal" in rmap
        assert rmap["tests.fixtures.pkg.Animal"] == "tests.fixtures.pkg.animals.Animal"
        assert "tests.fixtures.pkg.Dog" in rmap
        assert rmap["tests.fixtures.pkg.Dog"] == "tests.fixtures.pkg.animals.Dog"


# ---------------------------------------------------------------------------
# canonicalize_fqn  (Fix 1)
# ---------------------------------------------------------------------------

class TestCanonicalizeFqn:
    def test_no_reexport_map_passthrough(self):
        assert canonicalize_fqn("pkg.Animal.speak", {}) == "pkg.Animal.speak"

    def test_class_canonicalized(self):
        rmap = {"pkg.Animal": "pkg.animals.Animal"}
        assert canonicalize_fqn("pkg.Animal", rmap) == "pkg.animals.Animal"

    def test_method_on_reexported_class(self):
        rmap = {"pkg.Animal": "pkg.animals.Animal"}
        assert canonicalize_fqn("pkg.Animal.speak", rmap) == "pkg.animals.Animal.speak"

    def test_longer_prefix_wins(self):
        rmap = {
            "pkg.sub": "pkg.impl.sub",
            "pkg.sub.Cls": "pkg.impl.other.Cls",
        }
        # "pkg.sub.Cls.method" — longest matching prefix is "pkg.sub.Cls"
        assert canonicalize_fqn("pkg.sub.Cls.method", rmap) == "pkg.impl.other.Cls.method"

    def test_chained_reexport(self):
        # A → B → C
        rmap = {"a.Foo": "b.Foo", "b.Foo": "c.impl.Foo"}
        assert canonicalize_fqn("a.Foo.bar", rmap) == "c.impl.Foo.bar"

    def test_unrelated_fqn_unchanged(self):
        rmap = {"pkg.Animal": "pkg.animals.Animal"}
        assert canonicalize_fqn("other.module.func", rmap) == "other.module.func"


# ---------------------------------------------------------------------------
# enrich_star_imports  (Fix 2)
# ---------------------------------------------------------------------------

class TestEnrichStarImports:
    def test_within_repo_star_import_resolved(self):
        # Parse from project root so module FQNs match the absolute imports
        # in star_importer.py ("from tests.fixtures.star_exporter import *")
        PROJECT_ROOT = FIXTURES.parent.parent
        pf_exporter = parse_file(FIXTURES / "star_exporter.py", PROJECT_ROOT)
        pf_importer = parse_file(FIXTURES / "star_importer.py", PROJECT_ROOT)
        all_pfs = [pf_exporter, pf_importer]
        fqn_map = build_module_fqn_map(all_pfs)
        tables = {
            pf.file_path: build_symbol_table(pf, fqn_map[pf.file_path], fqn_map)
            for pf in all_pfs
        }
        enrich_star_imports(all_pfs, tables, fqn_map)

        importer_table = tables[pf_importer.file_path]
        # retry and TIMEOUT should now be in the importer's symbol table
        assert "retry" in importer_table.names
        assert importer_table.names["retry"].endswith("star_exporter.retry")

    def test_external_star_import_not_touched(self):
        from parser import ParsedFile, ImportRef
        pf = ParsedFile(
            file_path="mymod.py",
            functions=[], classes=[], calls=[], variable_refs=[],
            imports=[ImportRef(file_path="mymod.py", module="os.path",
                               alias=None, imported_name="*")],
        )
        fqn_map = {"mymod.py": "mymod"}
        table = build_symbol_table(pf, "mymod", fqn_map)
        enrich_star_imports([pf], {"mymod.py": table}, fqn_map)
        # Nothing should have been added (os.path not in our parsed set)
        assert "join" not in table.names

    def test_all_restricts_star_imports(self):
        # Fix D: if source module defines __all__, only those names should be added
        PROJECT_ROOT = FIXTURES.parent.parent
        pf_exp = parse_file(FIXTURES / "all_exporter.py", PROJECT_ROOT)
        pf_imp = parse_file(FIXTURES / "star_importer.py", PROJECT_ROOT)
        all_pfs = [pf_exp, pf_imp]
        # Override star_importer to import from all_exporter instead
        # (We test this via a synthetic ParsedFile)
        from parser import ParsedFile, ImportRef
        pf_test_imp = ParsedFile(
            file_path="test_imp.py",
            functions=[], classes=[], calls=[], variable_refs=[],
            exports=[], top_level_aliases=[],
            imports=[ImportRef(file_path="test_imp.py",
                               module="tests.fixtures.all_exporter",
                               alias=None, imported_name="*")],
        )
        fqn_map = {**build_module_fqn_map([pf_exp]), "test_imp.py": "test_imp"}
        tables = {
            pf_exp.file_path: build_symbol_table(pf_exp, fqn_map[pf_exp.file_path], fqn_map),
            "test_imp.py": build_symbol_table(pf_test_imp, "test_imp", fqn_map),
        }
        enrich_star_imports([pf_exp, pf_test_imp], tables, fqn_map)
        imp_table = tables["test_imp.py"]
        # PublicHelper should be added (it's in __all__)
        assert "PublicHelper" in imp_table.names
        # _InternalHelper must NOT be added (not in __all__)
        assert "_InternalHelper" not in imp_table.names

    def test_explicit_import_not_shadowed(self):
        from parser import ParsedFile, ImportRef, FunctionDef
        # Exporter has `retry`; importer does `from X import *` but also
        # has an explicit `from other import retry` — explicit wins.
        pf_exp = ParsedFile(
            file_path="exp.py",
            functions=[FunctionDef(name="retry", file_path="exp.py",
                                   start_line=1, end_line=2, docstring=None,
                                   is_method=False, class_name=None,
                                   qualname="retry", params=[], decorators=[],
                                   return_annotation=None)],
            classes=[], imports=[], calls=[], variable_refs=[],
        )
        pf_imp = ParsedFile(
            file_path="imp.py",
            functions=[], classes=[], calls=[], variable_refs=[],
            imports=[
                ImportRef(file_path="imp.py", module="other",
                          alias=None, imported_name="retry"),
                ImportRef(file_path="imp.py", module="exp",
                          alias=None, imported_name="*"),
            ],
        )
        fqn_map = {"exp.py": "exp", "imp.py": "imp"}
        tables = {
            pf.file_path: build_symbol_table(pf, fqn_map[pf.file_path], fqn_map)
            for pf in [pf_exp, pf_imp]
        }
        enrich_star_imports([pf_exp, pf_imp], tables, fqn_map)
        # Explicit import of `retry` from `other` must not be overwritten
        assert tables["imp.py"].names["retry"] == "other.retry"


# ---------------------------------------------------------------------------
# Fix E — assignment aliases in build_symbol_table (__init__.py)
# ---------------------------------------------------------------------------

class TestAssignmentAliasesInSymbolTable:
    def test_alias_resolved_via_import(self):
        # In __init__.py: `from .impl import _Cls; PublicCls = _Cls`
        # Symbol table should have PublicCls → pkg.impl._Cls
        PROJECT_ROOT = FIXTURES.parent.parent
        pf_init = parse_file(FIXTURES / "alias_init" / "__init__.py", PROJECT_ROOT)
        pf_impl = parse_file(FIXTURES / "alias_init" / "impl.py", PROJECT_ROOT)
        all_pfs = [pf_init, pf_impl]
        fqn_map = build_module_fqn_map(all_pfs)
        table = build_symbol_table(pf_init, fqn_map[pf_init.file_path], fqn_map)
        # ConcreteModel alias should resolve to the impl module's _ConcreteModel
        assert "ConcreteModel" in table.names
        assert table.names["ConcreteModel"] == "tests.fixtures.alias_init.impl._ConcreteModel"

    def test_alias_added_to_reexport_map(self):
        # The reexport map built from __init__.py should include ConcreteModel
        PROJECT_ROOT = FIXTURES.parent.parent
        pf_init = parse_file(FIXTURES / "alias_init" / "__init__.py", PROJECT_ROOT)
        pf_impl = parse_file(FIXTURES / "alias_init" / "impl.py", PROJECT_ROOT)
        all_pfs = [pf_init, pf_impl]
        fqn_map = build_module_fqn_map(all_pfs)
        rmap = build_reexport_map(all_pfs, fqn_map)
        # The ImportRef for _ConcreteModel should be in the reexport map
        assert "tests.fixtures.alias_init._ConcreteModel" in rmap or \
               any("ConcreteModel" in k for k in rmap)

    def test_non_init_aliases_not_in_symbol_table(self):
        # Aliases in non-__init__.py files should NOT be added to symbol table
        # (only __init__.py files get this treatment)
        from parser import ParsedFile, ImportRef
        pf = ParsedFile(
            file_path="regular.py",
            functions=[], classes=[], imports=[], calls=[], variable_refs=[],
            exports=[], top_level_aliases=[("Alias", "SomeClass")],
        )
        fqn_map = {"regular.py": "regular"}
        table = build_symbol_table(pf, "regular", fqn_map)
        # Alias should NOT appear (only processed for __init__.py)
        assert "Alias" not in table.names
