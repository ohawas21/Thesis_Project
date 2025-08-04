from _typeshed import Incomplete

def is_subclass_but_not_base(cls, base): ...

class InheritanceTracker(type):
    """InheritanceTracker.

    Metaclass mixin to track the class inheriting from a given class. A
    typical use-case is to generate automatically tests on a hierarchy
    of classes.
    """
    __inheritors__: Incomplete
    def __new__(cls, name, bases, dct): ...
