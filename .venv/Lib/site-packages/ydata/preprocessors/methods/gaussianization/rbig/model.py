from sklearn.base import BaseEstimator, TransformerMixin

from ydata.preprocessors.methods.gaussianization.rbig.config import RBIGConfig
from ydata.preprocessors.methods.gaussianization.rbig.training import train_rbig_info_loss


class RBIG(BaseEstimator, TransformerMixin):
    def __init__(self, config: RBIGConfig):
        """Initialize a RBIG model.

        Args:
            config (RBIGConfig): Model config

        Returns:
            FlowModel: RBIG model
        """
        self.config = config
        self.info_loss = None
        self.gf_model = None

    def fit(self, X, y=None, progress_bar: bool = False):

        gf_model = train_rbig_info_loss(
            X=X, config=self.config, progress_bar=progress_bar
        )
        self.gf_model = gf_model
        self.info_loss = gf_model.info_loss
        return self

    def transform(self, X, y=None):
        return self.gf_model.forward(X)

    def inverse_transform(self, X, y=None):
        return self.gf_model.inverse(X)

    # CLEANME: not being used (@quemy)
    def log_det_jacobian(self, X, y=None):
        return self.gf_model.gradient(X)

    def predict_proba(self, X, y=None):
        return self.gf_model.predict_proba(X)

    def sample(self, n_samples: int = 10):
        return self.gf_model.sample(n_samples)

    def total_correlation(self):
        return self.info_loss.sum()
