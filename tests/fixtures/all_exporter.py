"""Fixture: module with __all__ restriction — Fix D verification.

Only `PublicHelper` is in __all__. Star-importing this module should
only add PublicHelper to the importer's symbol table, NOT _InternalHelper.
"""

__all__ = ["PublicHelper"]


def PublicHelper():
    return "public"


def _InternalHelper():
    return "internal"
