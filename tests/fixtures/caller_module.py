"""Module that imports and calls functions from simple_module."""
from simple_module import standalone_function


def wrapper(val: float) -> float:
    """Wrap standalone_function."""
    return standalone_function(val)


def double_wrap(val: float) -> float:
    """Wrap wrapper."""
    return wrapper(val)
