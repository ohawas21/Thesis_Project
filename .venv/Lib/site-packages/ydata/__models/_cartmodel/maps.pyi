from _typeshed import Incomplete
from enum import Enum

class CART_FLAVOR(Enum):
    TAB = (0, 'Tabular')
    SEQ = (1, 'Sequential')

NAME_TO_FLAVOR: Incomplete
ENABLED_DATATYPES: Incomplete

class BaseMethods(Enum):
    """Default list of methods.

    These methods are not directly available to hierarchical flavors. Most
    methods should be similar between all hierarchical cart flavors. However, in
    case it cannot be the case for technical reasons, we can override
    the method with a flavor-specific one. This prevented regression as
    a bug introduced in a specific flavor operator will not be
    propagated to another flavor silently. It also allows for
    experimentation and side-by-side comparison of methods.
    """
    EMPTY = ...
    SAMPLE = ...
    CART = ...
    NORM = ...
    NORMRANK = ...
    POLYREG = ...
    PARAMETRIC = (6, None)
    PERTURB = ...
    SEQ_EMPTY = ...
    SEQ_CART = ...
    @property
    def id(self): ...
    @property
    def function(self): ...

DATATYPE_TO_FUNCTION: Incomplete
ENABLED_METHODS: Incomplete
METHODS_MAP: Incomplete
METHOD_TO_TYPE_TO_FUNCTION: Incomplete
INIT_METHODS_MAP: Incomplete
DEFAULT_METHODS_MAP: Incomplete
NA_METHODS: Incomplete

class Smoothing(Enum):
    NA = 'NA'
    DENSITY = 'density'
