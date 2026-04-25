"""Module with a 3-deep call chain for impact radius testing."""

def alpha():
    """Top of the chain."""
    beta()


def beta():
    """Middle of the chain."""
    gamma()


def gamma():
    """Bottom of the chain."""
    pass
