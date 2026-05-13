"""Fixture: super() call — Fix B verification.

super().speak() in Dog.fetch should resolve to Animal.speak via
the super() handler + inheritance resolver (Fix A + Fix B).
"""

from tests.fixtures.pkg.animals import Animal


class GuideDog(Animal):
    def speak(self):
        base = super().speak()    # Fix B: super() → Animal.speak via inheritance
        return f"Guide: {base}"

    def breathe(self):
        return super().breathe()  # Fix A+B: Animal.breathe
