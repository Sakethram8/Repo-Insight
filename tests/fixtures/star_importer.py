"""Fixture: star-imports from star_exporter and uses the imported names.

A CALLS edge should be created from run_with_retry → retry, and a READS
edge from run_with_retry → star_exporter for TIMEOUT.
This fixture verifies Fix 2 (within-repo star import resolution).
"""

from tests.fixtures.star_exporter import *  # noqa: F401, F403


def run_with_retry(fn):
    for _ in range(TIMEOUT):  # reads TIMEOUT from star_exporter
        result = retry(fn)    # calls retry from star_exporter
        if result:
            return result
    return None
