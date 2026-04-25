"""A simple module for testing parser extraction."""

class Calculator:
    """Performs basic arithmetic."""

    def add(self, a: int, b: int) -> int:
        """Return the sum of a and b."""
        return a + b

    def subtract(self, a: int, b: int) -> int:
        """Return a minus b."""
        return a - b


def standalone_function(x: float) -> float:
    """Square a number."""
    return x * x
