"""Fixture: module that exports constants and a function for star-import tests."""

TIMEOUT = 30
MAX_RETRIES = 3


def retry(fn):
    for _ in range(MAX_RETRIES):
        result = fn()
        if result:
            return result
    return None
