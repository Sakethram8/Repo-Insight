"""Fixture: imports and actively uses constants from config_fixture."""

from tests.fixtures.config_fixture import DATABASE_URL, TIMEOUT


def connect():
    """Uses DATABASE_URL — READS edge should be created."""
    return DATABASE_URL


def retry_with_timeout(fn):
    """Uses TIMEOUT — READS edge should be created."""
    for _ in range(TIMEOUT):
        result = fn()
        if result:
            return result
    return None
