"""Fixture: defines Animal and Dog — the canonical definition site."""


class Animal:
    def speak(self):
        return "..."

    def breathe(self):
        return "inhale"


class Dog(Animal):
    def speak(self):
        return "woof"
