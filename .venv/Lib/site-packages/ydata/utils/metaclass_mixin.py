from collections import defaultdict


def is_subclass_but_not_base(cls, base):
    return issubclass(cls, base) and cls != base


class InheritanceTracker(type):
    """InheritanceTracker.

    Metaclass mixin to track the class inheriting from a given class. A
    typical use-case is to generate automatically tests on a hierarchy
    of classes.
    """

    __inheritors__ = defaultdict(list)

    def __new__(cls, name, bases, dct):
        klass = type.__new__(cls, name, bases, dct)
        for base in klass.mro()[1:-1]:
            cls.__inheritors__[base].append(klass)
            base.inheritors = cls.__inheritors__[base]
        return klass
