"""Fixture: defines classes/functions to be imported by resolution_caller."""


class Animal:
    """Base animal class."""

    def speak(self):
        return "..."


class Dog(Animal):
    """Dog inherits from Animal."""

    def speak(self):
        return "Woof"


def standalone_helper(x):
    """A module-level helper function."""
    return x * 2
