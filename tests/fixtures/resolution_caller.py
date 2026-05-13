"""Fixture: imports from resolution_target and makes calls for resolver tests."""

from tests.fixtures.resolution_target import Animal, standalone_helper
import os.path as osp


class Cat(Animal):
    """Cat also inherits from Animal — base resolved via symbol table."""

    def speak(self):
        return "Meow"

    def process(self):
        # self.speak() → should resolve to tests.fixtures.resolution_caller.Cat.speak
        result = self.speak()
        # standalone_helper → should resolve to tests.fixtures.resolution_target.standalone_helper
        return standalone_helper(result)


def path_user():
    # osp.join → should resolve to os.path.join
    return osp.join("/tmp", "file.txt")
