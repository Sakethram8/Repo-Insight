"""Fixture: imports constants but never uses them — no READS edges should be created."""

from tests.fixtures.config_fixture import DATABASE_URL, TIMEOUT  # noqa: F401


def do_something():
    """Does not use DATABASE_URL or TIMEOUT at all."""
    return 42
