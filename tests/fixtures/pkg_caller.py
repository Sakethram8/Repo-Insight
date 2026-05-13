"""Fixture: imports Animal via pkg.__init__ (re-export), not pkg.animals directly.

A CALLS edge from make_animal_speak → Animal.speak should resolve to
tests.fixtures.pkg.animals.Animal.speak, NOT tests.fixtures.pkg.Animal.speak.
This fixture verifies Fix 1 (__init__.py re-export canonicalization).
"""

from tests.fixtures.pkg import Animal


def make_animal_speak(name: str) -> str:
    animal = Animal()
    return animal.speak()
