from _typeshed import Incomplete
from sklearn.base import BaseEstimator, TransformerMixin
from ydata.preprocessors.methods.gaussianization.rbig.config import RBIGConfig

class RBIG(BaseEstimator, TransformerMixin):
    config: Incomplete
    info_loss: Incomplete
    gf_model: Incomplete
    def __init__(self, config: RBIGConfig) -> None:
        """Initialize a RBIG model.

        Args:
            config (RBIGConfig): Model config

        Returns:
            FlowModel: RBIG model
        """
    def fit(self, X, y: Incomplete | None = None, progress_bar: bool = False): ...
    def transform(self, X, y: Incomplete | None = None): ...
    def inverse_transform(self, X, y: Incomplete | None = None): ...
    def log_det_jacobian(self, X, y: Incomplete | None = None): ...
    def predict_proba(self, X, y: Incomplete | None = None): ...
    def sample(self, n_samples: int = 10): ...
    def total_correlation(self): ...
