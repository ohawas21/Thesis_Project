"""Class that splits the data for purposes of creating the training and holdout
sets."""
from pickle import dump as pdump
from pickle import load as pload
from typing import Literal, Optional
from uuid import uuid4

from dask.dataframe import DataFrame

from ydata.dataset import Dataset
from ydata.metadata import Metadata
from ydata.utils.random import RandomSeed
from ydata.utils.sampling.random import RandomSplitSampler
from ydata.utils.sampling.stratified import StratifiedSampler


def check_unique_index(dataset: DataFrame):
    isunique = True
    index = dataset.index
    len_idx = len(index)
    if len(index.unique()) < len_idx:
        isunique = False
    return isunique


class Holdout:
    """Class responsible for selecting an holdout/test subset from a given
    Dataset. Stores the holdout indexes.

    Arguments:

    Properties:
        holdout_index (): Holdout index
    """

    def __init__(self, fraction: float = 0.2):
        """Initialize Holdout class."""
        self._fraction = fraction
        self._holdout_def = None
        self._train_def = None
        self.uuid = str(uuid4())

        if fraction < 0 or fraction > 1:
            raise ValueError(
                "Argument `fraction` must be strictly between 0 and 1")

    def get_split(
        self,
        X: Dataset,
        metadata: Optional[Metadata] = None,
        random_state: RandomSeed = None,
        strategy: Literal["random", "stratified"] = "random",
    ) -> tuple:
        """Returns the indexes of the holdout portion."""

        # TODO: The stratified strategy must be revised to handle non-monotonically increasing indexes.
        if strategy == "stratified":
            raise NotImplementedError(
                "The current implementation of the stratified strategy is outdated " +
                "and will not work properly with indexes that restart at 0 for each " +
                "partition. Please use the random strategy instead.")

        strategies = {
            "random": RandomSplitSampler(),
            "stratified": StratifiedSampler(),
        }

        holdout, train = strategies[strategy].sample(
            X, frac=self._fraction,
            metadata=metadata,
            return_non_sampled=True,
            random_state=random_state
        )

        self._holdout_def = (
            holdout._data.divisions
            if not all(map(lambda x: x is None, holdout._data.divisions))
            else None,
            holdout._data.npartitions,
        )
        self._train_def = (
            train._data.divisions
            if not all(map(lambda x: x is None, train._data.divisions))
            else None,
            train._data.npartitions,
        )

        self._data = holdout._data
        self._train_data = train._data

        return train, holdout

    def _get_holdout_data(self) -> Dataset:
        """Get the published holdout Dataset."""
        holdout = self._data

        return Dataset(
            holdout,
            index=holdout.index,
            divisions=holdout.divisions
            if not all(map(lambda x: x is None, holdout.divisions))
            else None,
        )

    def _get_train_data(self) -> Dataset:
        """Get the published holdout Dataset."""
        train = self._train_data

        return Dataset(
            train,
            index=train.index,
            divisions=train.divisions
            if not all(map(lambda x: x is None, train.divisions))
            else None,
        )

    @property
    def holdout_def(self):
        """Returns the holdout divisitions and npartitions to be loaded while
        loading the same Dataset again."""
        return self._holdout_def

    def save(self, path: str):
        """Saves the Holdout object."""
        with open(path, "wb") as f:
            pdump(self, f)
            f.close()

    @classmethod
    def load(cls, path: str):
        """Load the Holdout object."""
        with open(path, "rb") as f:
            holdout = pload(f)
        assert isinstance(
            holdout, Holdout
        ), "The loaded file must correspond to an Holdout object. Please validate the given input path."
        return holdout

    ###################
    # Dunder methods #
    ##################
    def __len__(self):
        return len(self.index(delayed=False))
