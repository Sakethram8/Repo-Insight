"""Fixture: nested classes and nested functions for parser/resolver tests."""


class Outer:
    """Outer class."""

    class Inner:
        """Nested inner class."""

        def method(self):
            """Method on inner class."""
            self.helper()

        def helper(self):
            return 42

    def outer_method(self):
        self.Inner()

        def local_fn():
            return 1

        return local_fn()
