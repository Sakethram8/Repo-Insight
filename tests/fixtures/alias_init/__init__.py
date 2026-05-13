"""Fixture: __init__.py that uses assignment alias — Fix E verification.

`ConcreteModel = _ConcreteModel` (assignment alias, not import) should
be detected and added to the reexport map.
"""

from tests.fixtures.alias_init.impl import _ConcreteModel

ConcreteModel = _ConcreteModel   # Fix E: assignment alias
