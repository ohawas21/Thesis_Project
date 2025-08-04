
from numpy import array as np_array
from numpy import clip as np_clip
from numpy import mean as np_mean
from numpy import ndarray
from numpy import round as np_round
from numpy import sum as np_sum
from numpy import unique as np_unique
from numpy import zeros as np_zeros
from numpy.random import choice, randint
from pandas import DataFrame as pdDataFrame
from pandas import concat as pd_concat
from scipy.stats import entropy
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import BallTree, NearestNeighbors

from ydata.report.logger import logger
from ydata.report.metrics import MetricType
from ydata.report.metrics.base_metric import BaseMetric
from ydata.report.styles import StyleHTML


def get_duplicates(df_real, df_fake):
    """Retrieve any observations from the synthetic dataset that exist in the
    real dataset.

    Returns:
        duplicates = A Dataframe containing the duplicated rows
    """
    real_hashes = df_real.apply(lambda x: hash(tuple(x)), axis=1)
    fake_hashes = df_fake.apply(lambda x: hash(tuple(x)), axis=1)
    dups_mask = fake_hashes.isin(real_hashes.values)
    duplicates = df_fake.loc[dups_mask]
    ratio = len(duplicates) / df_real.shape[0] if df_real.shape[0] > 0 else 0.
    return duplicates, round(ratio, 2)


class ExactMatch(BaseMetric):
    def __init__(self, formatter=StyleHTML) -> None:
        super().__init__(formatter)

    @staticmethod
    def _get_description(formatter):
        description = f"The {formatter.bold('EXACT MATCHES')} score counts the \
            percentage of sensitive records in the synthetic data that match the \
            records in the original dataset. It is bounded between [0-1]. The \
            score must be 0 for safe data-sharing."

        return description

    def _evaluate(self, source, synthetic, **kwargs):
        _, duplicates_ratio = get_duplicates(source, synthetic)
        return np_clip(duplicates_ratio, 0.0, 1.0)

    @property
    def name(self) -> str:
        return "Exact Matches"

    @property
    def type(self) -> MetricType:
        return MetricType.NUMERIC


class SynthClassifier(BaseMetric):
    def __init__(self, formatter=StyleHTML, exclude_entity_col: bool = True) -> None:
        super().__init__(formatter, exclude_entity_col)

    @staticmethod
    def _get_description(formatter):
        description = f"The {formatter.bold('SYNTHETIC CLASSIFIER')} score is the \
            ROC-AUC of a model trained to distinguish between real and synthetic data. \
            A score close to 1 indicates that the estimator is not able to discriminate \
            between the original and synthetic data records, making the data safer to be \
            shared. This metric is bounded between [0-1]."

        return description

    def _evaluate(self, source, synthetic, **kwargs):
        if kwargs["is_timeseries"]:
            score = self._timeseries_classifier(
                source, synthetic, kwargs["entity_data"])
        else:
            score = self._tabular_classifier(source, synthetic)

        # normalise score to 0-1:
        # privacy 1 for any score worse than random (roc_auc_score <= 0.5) and
        # normalised afterwards
        score = 1 - 2 * np_clip(score - 0.5, 0, 0.5)
        return score

    def _get_sample_sequence_start(self, source_length: int, sample_length: int) -> int:
        if source_length <= sample_length:
            return 0
        return randint(0, source_length - sample_length)

    def _pad_sequence(
        self, dataframe: pdDataFrame, sequence_length: int
    ) -> pdDataFrame:
        df_size = len(dataframe)
        if df_size < sequence_length:
            padding = pdDataFrame(
                np_zeros((sequence_length - df_size, len(dataframe.columns))),
                columns=dataframe.columns,
            )
            dataframe = pd_concat([dataframe, padding])
        return dataframe

    def _sample_sequences(
        self,
        dataframe: pdDataFrame,
        sample_size: int,
        sequence_length: int,
        entity_data: pdDataFrame = None,
    ):
        if not entity_data.empty:
            sequences = []
            entities = entity_data.apply(
                lambda x: tuple(x), axis=1).unique()
            for _ in range(sample_size):
                entity = choice(entities)
                entity_filtered_data = entity_data[(
                    entity_data == entity).all(axis=1)]
                df = dataframe.loc[entity_filtered_data.index].reset_index(
                    drop=True)
                seq_start = self._get_sample_sequence_start(
                    len(df), sequence_length)
                sequences.append(
                    df.loc[seq_start: (seq_start + sequence_length - 1)])

            sequences = pd_concat(
                [self._pad_sequence(seq, sequence_length) for seq in sequences]
            )
        else:
            seq_start = randint(
                0,
                len(dataframe) - sequence_length,
                sample_size
            )
            sequences = [
                dataframe.iloc[i: (i + sequence_length)]
                for i in seq_start
            ]
            sequences = pd_concat(sequences)

        return sequences

    def _timeseries_classifier(self, df_real, df_synth, entity_data):
        """Classify whether the data is real or synthetic. High score means
        ease of recognizing the synthetic data.

        Args:
            df_real: Dataframe of real data
            df_synth: Dataframe of synthesized data
        Returns:
            roc_auc_score = AUC Score of the model (classifying synthesized or real)
        """
        # Changes the type in case of a nparray or similar to not get errors
        # TODO check if is possible to receive anything other than dataframe
        if type(df_real) == ndarray:
            df_real = pdDataFrame(df_real)
        if type(df_synth) == ndarray:
            df_synth = pdDataFrame(df_synth)

        sample_size = 2000
        seq_length = 10
        sequences = pd_concat(
            [
                self._sample_sequences(
                    df_real, sample_size, seq_length, entity_data["real"]),
                self._sample_sequences(
                    df_synth, sample_size, seq_length, entity_data["synth"]),
            ]
        )
        n_features = len(sequences.columns)
        sequences = np_array(sequences.values).reshape(
            sample_size * 2, seq_length * n_features
        )

        labels = np_array([1] * sample_size + [0] * sample_size)

        scores = cross_val_score(X=sequences, y=labels, estimator=GradientBoostingClassifier(
        ), cv=StratifiedKFold(2), scoring='roc_auc')
        return np_mean(scores)

    def _tabular_classifier(self, df_real, df_synth):
        """Classify whether the data is real or synthetic. High score means
        ease of recognizing the synthetic data.

        Args:
            df_real: Dataframe of real data
            df_synth: Dataframe of synthesized data
        Returns:
            roc_auc_score = AUC Score of the model (classifying synthesized or real)
        """
        # Changes the type in case of a nparray or similar to not get errors

        if type(df_real) == ndarray:
            df_real = pdDataFrame(df_real)
        if type(df_synth) == ndarray:
            df_synth = pdDataFrame(df_synth)

        model = GradientBoostingClassifier()
        data = pd_concat((df_real, df_synth), axis=0).reset_index(drop=True)
        data["Fake"] = 0
        data.loc[df_real.shape[0]:, "Fake"] = 1
        train, test = train_test_split(data, test_size=0.3, random_state=42)
        model = model.fit(train.drop(["Fake"], axis=1), train["Fake"])
        preds = model.predict_proba(test.drop(["Fake"], axis=1))[:, 1]
        roc_auc = roc_auc_score(test["Fake"], preds)

        return roc_auc

    @property
    def name(self) -> str:
        return "Synthetic Classifier"


class MembershipDisclosureScore(BaseMetric):
    def __init__(self, formatter=StyleHTML, n_records: int = 1000):
        super().__init__(formatter)
        msg = "n_records must be `int`"
        assert isinstance(n_records, int), msg
        msg = "n_records must be positive"
        assert n_records > 0, msg

        self.n_records = n_records

    @property
    def name(self) -> str:
        return "Membership Inference Risk"

    @staticmethod
    def _get_description(formatter):
        description = f"The {formatter.bold('MEMBERSHIP INFERENCE SCORE')} score \
            measures the risk that an attacker can determine whether a particular \
            record of the original dataset was used to train the synthesizer. A \
            membership inference score close to 0 indicates that an attacker is \
            unlikely to determine if a specific record was a member of the original \
            dataset used to train the synthesizer. This metric is bounded between [0,1]."

        return description

    def _evaluate(self, source, synthetic, **kwargs):
        return self.score(source, synthetic, kwargs["training_data"])

    def score(self, holdout, synthetic, training, random_state=None):
        # sample n_records from holdout and training dfs
        n_records = min([self.n_records, len(holdout), len(training)])
        if n_records < self.n_records:
            logger.info(
                "[PROFILEREPORT] - Membership Disclosure Score sample "
                + f"size was reduce to match the dataset with size {n_records}."
            )

        # TODO: entity aware sampling
        holdout_sample = holdout.sample(n_records, random_state=random_state)
        training_sample = training.sample(n_records, random_state=random_state)
        training_sample = training_sample[holdout_sample.columns]

        # get labels and scores from the adversary
        labels = np_array([0] * n_records + [1] * n_records)
        scores = self.adversiarial_attack(
            pd_concat([holdout_sample, training_sample]), synthetic
        )

        # ROC AUC score
        score = roc_auc_score(labels, scores)

        # normalise score to 0-1:
        # privacy 1 for any score worse than random (roc_auc_score <= 0.5) and
        # normalised afterwards
        score = 2 * np_clip(score - 0.5, 0, 0.5)

        return score

    def adversiarial_attack(self, source, synthetic):
        # build BallTree using hamming distance and
        # query it to find closest synth record to every train and test record
        # TODO aggregate by entity? use another method?
        # TODO this may work: https://tslearn.readthedocs.io/en/stable/auto_examples/neighbors/plot_knn_search.html
        tree = BallTree(synthetic, metric="hamming")
        distances, _ = tree.query(source, k=1)
        # convert distances to probability scores
        scores = 1 - distances

        return scores


class IdentifiabilityScore(BaseMetric):
    def __init__(self, formatter=StyleHTML) -> None:
        super().__init__(formatter)

    @staticmethod
    def _get_description(formatter):
        description = f"The {formatter.bold('IDENTIFIABILITY')} score returns the \
            re-identification risk on the real dataset from the synthetic dataset. \
            It is bounded between [0-1]. A score of 0 represents no re-identification risk."
        return description

    def _evaluate(self, source, synthetic, **kwargs):
        """Adapted from https://github.com/vanderschaarlab/synthcity/blob/main/
        src/synthcity/metrics/eval_privacy.py."""
        np_source = source.to_numpy().reshape(len(source), -1)
        np_synthetic = synthetic.to_numpy().reshape(len(synthetic), -1)
        if kwargs.get("weighted", False):
            np_source = self._calculated_weighted_data(np_source)
            np_synthetic = self._calculated_weighted_data(np_synthetic)

        nbrs = NearestNeighbors(n_neighbors=2).fit(np_source)
        distance, _ = nbrs.kneighbors(np_source)

        nbrs_hat = NearestNeighbors(n_neighbors=1).fit(np_synthetic)
        distance_hat, _ = nbrs_hat.kneighbors(np_source)

        R_Diff = distance_hat[:, 0] - distance[:, 1]
        identifiability_value = np_sum(R_Diff < 0) / float(np_source.shape[0])
        return identifiability_value

    def _calculated_weighted_data(self, data: ndarray):
        assert isinstance(data, ndarray), "Non supported data type"
        weights = np_zeros([data.shape[1], ])
        for i in range(data.shape[1]):
            _, counts = np_unique(np_round(data[:, i]),
                                  return_counts=True)
            weights[i] = entropy(counts)
        weighted = data.copy()

        for i in range(data.shape[1]):
            if weights[i] != 0:
                weighted[:, i] = data[:, i] / weights[i]
            else:
                weighted[:, i] = data[:, i]
        return weighted

    @property
    def name(self) -> str:
        return "Identifiability Risk"

    @property
    def type(self) -> MetricType:
        return MetricType.NUMERIC
