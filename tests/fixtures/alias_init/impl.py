"""Fixture: defines the concrete class that alias_init.__init__ re-exports."""


class _ConcreteModel:
    def save(self):
        return "saved"

    def delete(self):
        return "deleted"
