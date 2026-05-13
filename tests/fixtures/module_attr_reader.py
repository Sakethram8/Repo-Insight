"""Fixture: reads constants via module attribute access — Fix C verification.

`import config_fixture` then `config_fixture.DATABASE_URL` should produce
a READS edge from read_url → tests.fixtures.config_fixture.
"""

import tests.fixtures.config_fixture as cfg


def read_url():
    return cfg.DATABASE_URL   # Fix C: compound VariableRef → READS edge


def read_timeout():
    return cfg.TIMEOUT         # Fix C: compound VariableRef → READS edge
