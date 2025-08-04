from _typeshed import Incomplete
from ydata.synthesizers.base_synthesizer import BaseSynthesizer

logger: Incomplete

class GMMSynthesizer(BaseSynthesizer):
    max_components: Incomplete
    def __init__(self, *, max_components: int = 5) -> None: ...
    model_: Incomplete
    def fit(self, X, y: Incomplete | None = None, **kwargs): ...
    def sample(self, n_samples: int = 1): ...
