from _typeshed import Incomplete

def skewness(col, data):
    """
    Parameters
    ----------
    col Name of the column to be transformed
    data series with the column values
    Returns A transformer function to correct skewness
    -------
    """

class MultiModalNumberTransformer:
    """Reversible transform for multimodal data.
    To effectively sample values from a multimodal distribution, we cluster values of a
    numerical variable using a `skelarn.mixture.GaussianMixture`_ model (GMM).
    Args:
        num_modes(int): Number of modes on given data.
    Attributes:
        num_modes(int): Number of components in the `skelarn.mixture.GaussianMixture`_ model.
    .. _skelarn.mixture.GaussianMixture: https://scikit-learn.org/stable/modules/generated/
        sklearn.mixture.GaussianMixture.html
    """
    num_modes: Incomplete
    model: Incomplete
    means: Incomplete
    stds: Incomplete
    def __init__(self, num_modes: int = 5) -> None:
        """Initialize instance."""
    def fit(self, data): ...
    def transform(self, data):
        """Cluster values using a `skelarn.mixture.GaussianMixture`_ model.
        Args:
            data(numpy.ndarray): Values to cluster in array of shape (n,1).
        Returns:
            tuple[numpy.ndarray, numpy.ndarray, list, list]: Tuple containg the features,
            probabilities, averages and stds of the given data.
        .. _skelarn.mixture.GaussianMixture: https://scikit-learn.org/stable/modules/generated/
            sklearn.mixture.GaussianMixture.html
        """
    def inverse_transform(self, data):
        """Reverse the clustering of values.

        Args:
            data(numpy.ndarray): Transformed data to restore.
            info(dict): Metadata.
        Returns:
           numpy.ndarray: Values in the original space.
        """
