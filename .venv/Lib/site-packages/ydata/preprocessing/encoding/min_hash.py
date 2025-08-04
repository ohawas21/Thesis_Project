"""Categorical variables encoder based on hashing Best approach for high-
cardinality variables. Can be also used to detect and mitigate issues of
categorical mistakes/errors (dirty categories)

Code adapted from: https://github.com/dirty-cat/dirty_cat/blob/master/dirty_cat/minhash_encoder.py
"""

from numpy import array, asarray, concatenate, infty, minimum, ones, zeros
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import murmurhash3_32

from ._fast_hash import ngram_min_hash
from .utils import LRUDict

# This function can be leverage by other methods


def check_input(X):
    """Check input data shape.

    Also converts X to a numpy array if not already.
    """
    X = asarray(X)
    if X.ndim != 2:
        raise ValueError(
            "Expected 2D array. Reshape your data either using"
            "array.reshape(-1, 1) if your data has a single feature or"
            "array.reshape(1, -1) if it contains a single sample."
        )
    return X


class MinHashEncoder(BaseEstimator, TransformerMixin):
    """
    Encode string categorical features as a numeric array, minhash method
    applied to ngram decomposition of strings based on ngram decomposition
    of the string.
    Parameters
    ----------
    n_components : int, default=30
        The number of dimension of encoded strings. Numbers around 300 tend to
        lead to good prediction performance, but with more computational cost.
    ngram_range : tuple (min_n, max_n), default=(2, 4)
        The lower and upper boundary of the range of n-values for different
        n-grams to be extracted. All values of n such that min_n <= n <= max_n.
        will be used.
    hashing : str {'fast', 'murmur'}, default=fast
        Hashing function. fast is faster but
        might have some concern with its entropy.
    minmax_hash : bool, default=False
        if True, return min hash and max hash concatenated.
    handle_missing : 'error' or 'zero_impute' (default)
        Whether to raise an error or encode missing values (NaN) with
        vectors filled with zeros.
    References
    ----------
    For a detailed description of the method, see
    `Encoding high-cardinality string categorical variables
    <https://hal.inria.fr/hal-02171256v4>`_ by Cerda, Varoquaux (2019).
    """

    def __init__(
        self,
        n_components=30,
        ngram_range=(2, 4),
        hashing="fast",
        minmax_hash=False,
        handle_missing="zero_impute",
    ):
        self.ngram_range = ngram_range
        self.n_components = n_components
        self.hashing = hashing
        self.minmax_hash = minmax_hash
        self.count = 0
        self.handle_missing = handle_missing
        self._capacity = 2**10

    def get_unique_ngrams(self, string, ngram_range):
        """Return the set of unique n-grams of a string.
        Parameters
        ----------
        string : str
            The string to split in n-grams.
        ngram_range : tuple (min_n, max_n)
        The lower and upper boundary of the range of n-values for different
        n-grams to be extracted. All values of n such that min_n <= n <= max_n.
        Returns
        -------
        set
            The set of unique n-grams of the string.
        """
        spaces = " "  # * (n // 2 + n % 2)
        string = spaces + " ".join(string.lower().split()) + spaces
        ngram_set = set()
        for n in range(ngram_range[0], ngram_range[1] + 1):
            string_list = [string[i:] for i in range(n)]
            ngram_set |= set(zip(*string_list))
        return ngram_set

    def minhash(self, string, n_components, ngram_range):
        """Encode a string using murmur hashing function.
        Parameters
        ----------
        string : str
            The string to encode.
        n_components : int
            The number of dimension of encoded string.
        ngram_range : tuple (min_n, max_n)
        The lower and upper boundary of the range of n-values for different
        n-grams to be extracted. All values of n such that min_n <= n <= max_n.
        Returns
        -------
        array, shape (n_components, )
            The encoded string.
        """
        min_hashes = ones(n_components) * infty
        grams = self.get_unique_ngrams(string, self.ngram_range)
        if len(grams) == 0:
            grams = self.get_unique_ngrams(" Na ", self.ngram_range)
        for gram in grams:
            hash_array = array(
                [
                    murmurhash3_32("".join(gram), seed=d, positive=True)
                    for d in range(n_components)
                ]
            )
            min_hashes = minimum(min_hashes, hash_array)
        return min_hashes / (2**32 - 1)

    def get_fast_hash(self, string):
        """
        Encode a string with fast hashing function.
        fast hashing supports both min_hash and minmax_hash encoding.
        Parameters
        ----------
        string : str
            The string to encode.
        Returns
        -------
        array, shape (n_components, )
            The encoded string, using specified encoding scheme.
        """
        if self.minmax_hash:
            return concatenate(
                [
                    ngram_min_hash(string, self.ngram_range,
                                   seed, return_minmax=True)
                    for seed in range(self.n_components // 2)
                ]
            )
        else:
            return array(
                [
                    ngram_min_hash(string, self.ngram_range, seed)
                    for seed in range(self.n_components)
                ]
            )

    def fit(self, X, y=None):
        """
        Fit the MinHashEncoder to X. In practice, just initializes a dictionary
        to store encodings to speed up computation.
        Parameters
        ----------
        X : array-like, shape (n_samples, ) or (n_samples, 1)
            The string data to encode.
        Returns
        -------
        self
            The fitted MinHashEncoder instance.
        """
        self.hash_dict = LRUDict(capacity=self._capacity)
        return self

    def transform(self, X):
        """Transform X using specified encoding scheme.
        Parameters
        ----------
        X : array-like, shape (n_samples, ) or (n_samples, 1)
            The string data to encode.
        Returns
        -------
        array, shape (n_samples, n_components)
            Transformed input.
        """
        X = check_input(X)
        if self.minmax_hash:
            assert (
                self.n_components % 2 == 0
            ), "n_components should be even when minmax_hash=True"
        if self.hashing == "murmur":
            assert not (
                self.minmax_hash), "minmax_hash not implemented with murmur"
        if self.handle_missing not in ["error", "zero_impute"]:
            template = (
                "handle_missing should be either 'error' or " "'zero_impute', got %s"
            )
            raise ValueError(template % self.handle_missing)
        X_out = zeros((len(X), self.n_components))
        X = X.reshape(-1)

        # TODO Parallel run here
        nan_idx = []

        if self.hashing == "fast":
            for i, x in enumerate(X):
                if isinstance(x, float):  # true if x is a missing value
                    nan_idx.append(i)
                elif x not in self.hash_dict:
                    X_out[i, :] = self.hash_dict[x] = self.get_fast_hash(x)
                else:
                    X_out[i, :] = self.hash_dict[x]

        elif self.hashing == "murmur":
            for i, x in enumerate(X):
                if isinstance(x, float):
                    nan_idx.append(i)
                elif x not in self.hash_dict:
                    X_out[i, :] = self.hash_dict[x] = self.minhash(
                        x, n_components=self.n_components, ngram_range=self.ngram_range
                    )
                else:
                    X_out[i, :] = self.hash_dict[x]

        else:
            raise ValueError(
                "hashing function must be 'fast' or"
                "'murmur', got '{}'"
                "".format(self.hashing)
            )

        if self.handle_missing == "error" and nan_idx:
            msg = (
                "Found missing values in input data; set "
                "handle_missing='' to encode with missing values"
            )
            raise ValueError(msg)

        return X_out
