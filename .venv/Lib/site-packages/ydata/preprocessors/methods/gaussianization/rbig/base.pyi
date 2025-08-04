from _typeshed import Incomplete
from scipy.stats._multivariate import multi_rv_generic as multi_rv_generic

class Bijector:
    """Base class for a Bijector Mixin.

    A bijector is by definition fully invertible mapping with a
    gradient.
    """
    def forward(self, X) -> None: ...
    def inverse(self, X) -> None: ...
    def gradient(self, X) -> None: ...

class CompositeBijector(Bijector):
    """A class to compose bijectors.

    It is itself a Bijector by definition of composition.
    """
    bijectors: Incomplete
    def __init__(self, bijectors: list[Bijector]) -> None:
        """Composite Bijector initialization.

        Args:
            bijectors (List[Bijector]): Ordered list of bijector to compose
        """
    def forward(self, X): ...
    def inverse(self, Z): ...
    def gradient(self, X): ...

class FlowModel(CompositeBijector):
    """Base Flow Model class.

    A flow model is a generative model using an explicit likelihood
    function defined by the composition of many change-of-variable
    bijective transformations to transform a distribution into another.
    """
    base_dist: Incomplete
    info_loss: Incomplete
    def __init__(self, bijectors: list[Bijector], base_dist) -> None:
        """Flow Model initialization.

        Args:
            bijectors (List[Bijector]): Ordered list of bijector to compose
            base_dist (multi_rv_generic): Initial distribution to sample
        """
    def sample(self, n_samples: multi_rv_generic = 10): ...
    def predict_proba(self, X): ...
    def score_samples(self, X): ...
