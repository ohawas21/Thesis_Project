"""Implementation of a storage for Synthetic Data Fidelity, Utility and Privacy
metrics."""
from datetime import datetime

from dask.dataframe import DataFrame as ddDataFrame
from numpy import nan
from pandas import DataFrame as pdDataFrame
from pandas import Index as pdIndex
from pandas import Series as pdSeries
from pandas import concat as pd_concat
from pandas import to_datetime as pd_to_datetime
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from ydata.dataset import Dataset
from ydata.metadata import Metadata
from ydata.report.logger import logger
from ydata.report.metrics._dimension_reduction import DimensionReduction
from ydata.report.metrics._metrics import Autocorrelation, DistanceCorrelation, DistanceDistribution
from ydata.report.metrics._privacy import ExactMatch, IdentifiabilityScore, MembershipDisclosureScore, SynthClassifier
from ydata.report.metrics._utility import TSTR, FeatureImportance, QScore, TSTRTimeseries
from ydata.report.metrics.coverage import CategoryCoverage, MissingCategoryCoverage, RangeCoverage
from ydata.report.metrics.distribution import KSTest, TotalVariationDistance
from ydata.report.metrics.group import MetricGroup
from ydata.report.metrics.missing import MissingValuesSimilarity
from ydata.report.metrics.mutual_info import MutualInformationMatrix
from ydata.report.metrics.statistical import StatisticalSimilarity
from ydata.report.metrics.utils import is_entity
from ydata.report.reports.report_type import ReportType
from ydata.utils.data_types import CATEGORICAL_DTYPES, DataType, validate_datatypes


# @typechecked
class SyntheticDataMetrics:
    """Storage class for for Synthetic Data Fidelity, Utility and Privacy
    metrics.

    Args:
        report_type (ReportType): Whether to calculate the metrics for 'tabular' or 'timeseries' data. Defaults to 'tabular'.
        real: Dataset object with the real data to be evaluated
        synth: Dataset object with the synthetic data to be evaluated
    """

    def __init__(
        self,
        real: Dataset,
        synth: Dataset,
        data_types: dict,
        anonymized_cols: list[str],
        safe_mode: bool = True,
        training_data: Dataset | None = None,
        target: str | None = None,
        report_type: ReportType | str = ReportType.TABULAR,
        metadata: Metadata = None
    ):
        try:
            report_type = ReportType(report_type)
        except BaseException:
            raise Exception(
                f"The provided report_type ({report_type}) is not valid.")

        self.creation_date = datetime.utcnow()
        self._is_timeseries = report_type == ReportType.TIMESERIES

        self.target = target
        self.data_types = data_types
        self.metadata = metadata
        self.anonymized_cols = anonymized_cols
        self.safe_mode = safe_mode
        self._real = real
        self._synth = synth
        self._training_data = training_data
        self.stats_summary = {}

        # Validate input data
        self.__prevalidate()

        self.non_anonymized_cols = [
            c
            for c in self.metadata.columns.keys()
            if c not in self.anonymized_cols and c in self._real.columns and not is_entity(c, self.metadata)
        ]

        self.entity_cols = [
            c for c in metadata.dataset_attrs.entities if c in self._real.columns] if metadata.dataset_attrs else []

        self._preprocessed = False

    def __convert_datasets_to_pandas(self):
        self._real = self._real.to_pandas()
        self._synth = self._synth.to_pandas()
        if self._training_data is not None:
            self._training_data = self._training_data.to_pandas()

    def __sort_datasets(self):
        if self.metadata.dataset_attrs:
            self._real.sort_values(
                by=self.metadata.dataset_attrs.sortbykey, ignore_index=True, inplace=True
            )
            self._synth.sort_values(
                by=self.metadata.dataset_attrs.sortbykey, ignore_index=True, inplace=True
            )

            if self._training_data is not None:
                self._training_data.sort_values(
                    by=self.metadata.dataset_attrs.sortbykey, ignore_index=True, inplace=True
                )

    def __exclude_unsupported_columns(self):
        # TODO: Add requirement for timeseries DatetimeIndex as index
        # Check provided if provided dict with data types is valid
        valid_columns = validate_datatypes(
            self.data_types, valid_dtypes=[
                DataType.NUMERICAL, *CATEGORICAL_DTYPES, DataType.DATE]
        )
        all_null_cols = self._real.apply(lambda s: all(s.isna())).to_dict()
        all_null_cols = {k for k, v in all_null_cols.items() if v}
        excluded_columns = set(self._real.columns) - set(valid_columns)
        excluded_columns |= all_null_cols

        valid_columns = [
            col
            for col in valid_columns
            if col not in all_null_cols and col in self._synth.columns
        ]

        if excluded_columns:
            logger.warning(
                f"[PROFILEREPORT] - Found variables with data types not supported for synthetic metrics calculation: {excluded_columns}"
            )

        # Exclude invalid columns
        self._real = self._real[valid_columns]
        self._synth = self._synth[valid_columns]
        if self._training_data is not None:
            self._training_data = self._training_data[valid_columns]

        if len(valid_columns) == 0:
            raise Exception(
                "Please verify the provided input dataset. "
                "No column provided can be selected due to invalid data types, "
                "or there is no intersection between the columns from real and synthetic data."
                "The dataset must have at least a column with one of the following data types "
                f"[{DataType.STR.value},{DataType.CATEGORICAL.value},{DataType.NUMERICAL.value}]"
            )

    def __prevalidate(self, errors: str = "warn"):
        """Run pre-validations that guarantee downstream execution of the metrics calculations.
        Args:
            errors (str): default behaviour for dealing with prevalidation errors. Defaults to warn.
                - 'raise': stops the execution.
                - 'warn': logs the error as a warning and actively implements necessary data processing fixes.
        Checks:
        1. Whichever the input type, convert to Pandas DataFrame
        t1. If time series, sort rows
        2. Type check (double-failure)
        3. If target is specified, it must exist in both dataframes.
        """
        ERROR_LEVELS = ["warn", "raise"]
        assert (
            errors in ERROR_LEVELS
        ), f"Specified 'errors' argument ({errors}) is not supported. Supported: {ERROR_LEVELS}."

        self.__convert_datasets_to_pandas()
        if self._is_timeseries:
            self.__sort_datasets()

        self.__exclude_unsupported_columns()
        self.__adjust_synth_to_holdout()
        logger.info(
            f"[PROFILEREPORT] - Synthetic data quality report selected target variable: {self.target}"
        )

    def __adjust_synth_to_holdout(self):
        """This method ensures that the synthetic data does not have data
        outside of the holdout domain.

        In particular, the categories not available on the holdout are
        removed from the synthetic data, and the numerical columns are
        clipped on the synthetic data according to their domain in the
        holdout. Anonymized and unique-value columns are not to be
        updated.
        """

        def _filter_categories(df: pdDataFrame, cat_col: str, unique_vals: pdSeries):
            cat_inters = df[cat_col].isin(unique_vals)
            if any(cat_inters):
                return df[cat_inters]
            return df

        def _filter_range(df: pdDataFrame, num_col: str, col_min: float, col_max: float):
            return df[(df[num_col] >= col_min) & (df[num_col] <= col_max) | df[num_col].isna()]

        if self.anonymized_cols is None:
            self.anonymized_cols = []

        unique_cols = [
            uniq.column for uniq in self.metadata.warnings["unique"]]
        categorical_vars = [
            c for c in self._real.columns
            if c in self.metadata.categorical_vars
        ]
        numerical_vars = [
            c for c in self._real.columns
            if c in self.metadata.numerical_vars
        ]

        synth_cp = self._synth.copy()
        real_cp = self._real.copy()

        for cat_col in categorical_vars:
            if (
                cat_col not in self.anonymized_cols
                and cat_col not in unique_cols
                and not is_entity(cat_col, self.metadata)
            ):
                synth_cp = _filter_categories(
                    synth_cp, cat_col, real_cp[cat_col].unique())
                real_cp = _filter_categories(
                    real_cp, cat_col, synth_cp[cat_col].unique())

        for num_col in numerical_vars:
            if (
                num_col not in self.anonymized_cols
                and num_col not in unique_cols
                and not is_entity(num_col, self.metadata)
            ):
                synth_cp = _filter_range(
                    synth_cp, num_col,
                    real_cp[num_col].min(),
                    real_cp[num_col].max()
                )

        # This adjustment logic is only applied if both dataframes don't end up with less than 2 rows.
        if len(real_cp.index) >= 2 and len(synth_cp.index) >= 2:
            self._synth = synth_cp
            self._real = real_cp

    def __prep_data_format(
        self,
        dataframes: list[pdDataFrame | ddDataFrame],
        data_types: dict,
        metadata: Metadata | None
    ) -> tuple[list[pdDataFrame | ddDataFrame], dict, Metadata]:
        logger.info("[PROFILEREPORT] - preparing data format.")

        self.stats_summary["missing_values"] = {
            "real": dataframes[0].isna().sum(),
            "synth": dataframes[1].isna().sum()
        }
        # Check wether the id column exists and drop it
        for column_name, column_type in data_types.items():
            if is_entity(column_name, metadata):
                continue
            # ignore excluded columns
            if column_name not in dataframes[0].columns:
                continue
            elif DataType(column_type) in CATEGORICAL_DTYPES:
                for df in dataframes:
                    df[column_name] = df[column_name].astype("category")
                    # FIXME this code is unreachable since dataframes are converted to pandas prior to this
                    if isinstance(df, ddDataFrame):
                        df[column_name] = df[column_name].cat.as_known()
            elif DataType(column_type) == DataType.NUMERICAL:
                for df in dataframes:
                    if df.dtypes[column_name] == "object":
                        df[column_name].replace(
                            {"": nan, "?": nan}, inplace=True)
                    df[column_name] = df[column_name].astype("float")
                    df[column_name] = df[column_name].fillna(
                        df[column_name].quantile(0.5)
                    )
                    if column_type == "int":
                        df[column_name] = df[column_name].astype(column_type)
            elif DataType(column_type) == DataType.DATE:
                for df in dataframes:
                    df[column_name] = pd_to_datetime(
                        df[column_name]).view("int64") // 10**9
                data_types[column_name] = DataType.NUMERICAL
                # This is not the proper way to update the data type in the metadata.
                # However, the update method fails because of the variable type.
                # Updating the datatype directly is enough for the report.
                metadata.columns[column_name].datatype = DataType.NUMERICAL
        return dataframes, data_types, metadata

    def __update_inconsistent_dtypes(self, dataframes: list[pdDataFrame | ddDataFrame]):
        """Due to the anonymization procedure, the synthetic data may have
        columns whose data types don't match the original data.

        In this case, we try to convert the synthetic data type to the
        original one. If not possible, the column is deemed invalid
        since there is a mismatch between types, which prevents the
        calculation of the metrics.
        """
        invalid_colums = []
        original_dtypes = dataframes[0].dtypes
        for col, dtype in original_dtypes.items():
            try:
                for i in range(1, len(dataframes)):
                    dataframes[i][col] = dataframes[i][col].astype(dtype)
            except ValueError:
                if col not in self.metadata.dataset_attrs.entities:
                    invalid_colums.append(col)

        for i in range(len(dataframes)):
            dataframes[i] = dataframes[i].drop(columns=invalid_colums)
        self.metadata = self.metadata[list(dataframes[0].columns)]
        return dataframes

    def __prep_data_for_metrics(self,
                                dataframes: list[pdDataFrame | ddDataFrame],
                                data_types: dict,
                                metadata: Metadata | None
                                ) -> list[pdDataFrame | ddDataFrame]:
        """Data preparation for metrics report generation. The data processing
        is rather simple when compared to the one used for the synthetic data
        generation. For categorical, assumed an ordinal encoding despite the
        nature of the category. For numerical encoding, missing values are
        imputed with median and standartized afterwards. No further processing
        to correct data behaviours is done.

        :param dataframes: list of pandas dataframe
        :param data_types: dict with the dataframe data types
        :return: dataframes ready for metrics calculation
        """
        logger.info(
            "[PROFILEREPORT] - Preparing the data for metrics calculation")

        valid_colums = set(dataframes[0].columns)

        num_cols = [
            col for col in data_types
            if DataType(data_types[col]) == DataType.NUMERICAL and not is_entity(col, metadata) and col in valid_colums
        ]
        cat_cols = [
            col
            for col in data_types
            if DataType(data_types[col]) in CATEGORICAL_DTYPES and not is_entity(col, metadata) and col in valid_colums
        ]

        empty_cols = set()
        # If missing values, input the variable with the median
        for df in dataframes:
            df[num_cols] = df[num_cols].fillna(df[num_cols].quantile(0.5))
        for df in dataframes:
            na_str_cols = df.columns[df.eq('<NA>').any()].tolist()
            for col in na_str_cols:
                df[col] = df[col].replace('<NA>', nan)
            na_cols = df.columns[df.isna().any()].tolist()
            missing_constant = 99999999
            for col in na_cols:
                column = df[col].dropna()
                if len(column) == 0:
                    if is_entity(col, self.metadata):
                        df[col] = df[col].fillna(missing_constant)
                    else:
                        empty_cols.add(col)
                else:
                    first_element = column.iloc[0]
                    if isinstance(first_element, str):
                        missing_constant = "NA"
                    elif isinstance(first_element, float):
                        missing_constant = 99999999.0
                    if df[col].dtype == "category":
                        df[col] = (
                            df[col]
                            .cat.add_categories(missing_constant)
                            .fillna(missing_constant)
                        )
                    else:
                        df[col] = df[col].fillna(missing_constant)

        if len(empty_cols) > 0:
            for df in dataframes:
                cols_to_drop = [col for col in empty_cols if col in df.columns]
                df.drop(columns=cols_to_drop, inplace=True)
            self.non_anonymized_cols = [
                col for col in self.non_anonymized_cols if col not in empty_cols]

        num_cols = [c for c in num_cols if c not in empty_cols]
        cat_cols = [c for c in cat_cols if c not in empty_cols]
        columns = num_cols + cat_cols
        transformer = ColumnTransformer(
            [
                ("scale", StandardScaler(), pdIndex(num_cols)),
                ("ordinal", OrdinalEncoder(), pdIndex(cat_cols)),
            ]
        )
        transformer.fit(pd_concat(dataframes))
        for df in dataframes:
            df[columns] = transformer.transform(df[columns])

        non_supported_cols = [
            col for col in data_types if col not in columns and not is_entity(col, self.metadata)]
        if non_supported_cols:
            non_supported_types = {data_types[col]
                                   for col in non_supported_cols}
            logger.info(
                f"[PROFILEREPORT] - The columns {non_supported_cols} are not being considered for the metrics calculation."
                f" Either [{non_supported_types}] are not supported data types, the columns are empty or they have inconsistent "
                "types between the original and synthetic data."
            )

        return dataframes

    @staticmethod
    def _is_univariate(df: pdDataFrame):
        return len(df.columns) == 1

    def get_utility_metrics(
        self,
        real: pdDataFrame,
        synth: pdDataFrame,
        training_data: pdDataFrame | None = None,
    ):
        """Get utility metrics.

        - TSTR
        - TSTR for time-series
        - Discriminator
        """
        metrics = dict()
        real_ = real.copy()
        synth_ = synth.copy()

        utility = {}
        utility["qscore"] = QScore()

        if self.target:
            if self._is_timeseries:
                utility["tstr"] = TSTRTimeseries(exclude_entity_col=False)
            else:
                utility["tstr"] = TSTR()
                utility["feature_importance"] = FeatureImportance(
                    include_plot=True)

        utility = MetricGroup(utility, safe_mode=self.safe_mode)
        metrics = utility.evaluate(
            real_[self.non_anonymized_cols],
            synth_[self.non_anonymized_cols],
            data_types=self.data_types,
            metadata=self.metadata[self.non_anonymized_cols],
            target=self.target,
            entity_data={"real": real_[self.entity_cols],
                         "synth": synth_[self.entity_cols]}
        )
        return metrics

    def get_fidelity_metrics(
        self,
        real: pdDataFrame,
        synth: pdDataFrame,
        training_data: pdDataFrame | None = None,
    ):
        """Univariate consistency metrics.

        - Statistical distribution and tests
        - Distribution plots for visual validation
        - Features autocorrelation (specific for time-series)
        Global consistency metrics
        - Correlation matrix distances
        - PCA and UMAP 2D plots for visual validation
        - Mutual information (specific for time-series)
        """
        metrics = dict()
        real_ = real.copy()
        synth_ = synth.copy()

        fidelity = {}

        fidelity["distance_correlation"] = DistanceCorrelation()
        fidelity["distance_distribution"] = DistanceDistribution()
        fidelity["statistical_similarity"] = StatisticalSimilarity()
        fidelity["category_coverage"] = CategoryCoverage()
        fidelity["missing_category_coverage"] = MissingCategoryCoverage()
        fidelity["range_coverage"] = RangeCoverage()
        fidelity["ks_test"] = KSTest()
        fidelity["total_variation_distance"] = TotalVariationDistance()
        fidelity["missing_values_similarity"] = MissingValuesSimilarity()
        fidelity["mutual_information"] = MutualInformationMatrix(
            include_plots=True)

        if not self._is_univariate(real):
            fidelity["pca_chart"] = DimensionReduction()

        fidelity["synth_classifier"] = SynthClassifier(
            exclude_entity_col=not self._is_timeseries)

        if self._is_timeseries:
            fidelity["autocorrelation"] = Autocorrelation(
                exclude_entity_col=False)

        fidelity = MetricGroup(fidelity, safe_mode=self.safe_mode)
        metrics = fidelity.evaluate(
            real_[self.non_anonymized_cols],
            synth_[self.non_anonymized_cols],
            data_types=self.data_types,
            metadata=self.metadata[self.non_anonymized_cols],
            is_timeseries=self._is_timeseries,
            stats_summary=self.stats_summary,
            entity_data={"real": real_[self.entity_cols],
                         "synth": synth_[self.entity_cols]}
        )
        return metrics

    def _get_privacy_metrics_group(self):
        metrics = {
            "exact_matches": ExactMatch(),
            "identifiability": IdentifiabilityScore()
        }
        if self._training_data is not None:
            metrics["membership"] = MembershipDisclosureScore()
        return MetricGroup(metrics, safe_mode=self.safe_mode)

    def get_privacy_metrics(
        self,
        real: pdDataFrame,
        synth: pdDataFrame,
        training_data: pdDataFrame | None = None,
    ):
        """Calculate privacy metrics."""
        privacy = self._get_privacy_metrics_group()
        metrics = privacy.evaluate(
            real,
            synth,
            training_data=training_data,
            data_types=self.data_types,
            metadata=self.metadata,
            is_timeseries=self._is_timeseries,
        )
        return metrics

    def get_info_metrics(self):
        """Calculates general info metrics."""
        metrics = dict()
        metrics["nrows_real"] = self._real.shape[0]
        metrics["nrows_synth"] = self._synth.shape[0]
        metrics["columns_details"] = len(self.data_types)
        return metrics

    def get_anonymized_metrics(self):
        """Obtains info about the anonymized columns."""
        metrics = dict()
        metrics["n_anonymized_cols"] = len(self.anonymized_cols)
        metrics["anonymized_cols_details"] = None

        if len(self.anonymized_cols) > 0:
            details = []
            for col in self.anonymized_cols:
                col_metadata = self.metadata.columns[col]
                col_info = {
                    "Feature": col,
                    "Data Type": col_metadata.datatype.value,
                    "Variable Type": col_metadata.vartype.value,
                    "PII Type": ','.join([c.value for c in col_metadata.characteristics])
                }
                details.append(col_info)
            metrics["anonymized_cols_details"] = pdDataFrame.from_dict(details)

        return metrics

    def get_percentage_failed_metrics(self, privacy: dict, fidelity: dict, utility: dict):
        total_num_metrics = len(privacy) + len(fidelity) + len(utility)
        total_failed_metrics = len([m for m in privacy.values() if m.is_error]) + \
            len([m for m in fidelity.values() if m.is_error]) + \
            len([m for m in utility.values() if m.is_error])
        return total_failed_metrics / total_num_metrics

    def get_percentage_failed(self, metrics: dict):
        return len([m for m in metrics.values() if m.is_error]) / len(metrics)

    def get_error_logs(self, metrics: dict):
        return {m.name: f"[{m.name}] {str(m.values)}" for m in metrics.values() if m.is_error}

    def evaluate(self):
        """Calculate metrics for synthetic data quality between provided real
        and synth datasets.

        Returns:
            calculated metrics (dict)
        """
        # metrics will contain dicts of metrics per section (info, utility, fidelity, privacy)
        metrics = dict()
        metrics["info"] = self.get_info_metrics()
        metrics["anonymized"] = self.get_anonymized_metrics()

        # Converting to pandas as off today we calculate all the metrics do not leverage Dask or distributed processing
        dataframes = [self._real, self._synth]
        if self._training_data is not None:
            dataframes.append(self._training_data)

        if self._preprocessed is False:
            dataframes = self.__update_inconsistent_dtypes(dataframes)

            dataframes, self.data_types, self.metadata = self.__prep_data_format(
                dataframes, data_types=self.data_types, metadata=self.metadata)

            # Fazer aqui o preprocessing standard for the metrics
            dataframes = self.__prep_data_for_metrics(
                dataframes, data_types=self.data_types, metadata=self.metadata)

            self._preprocessed = True

        # Compute and store the multiple metrics per metric type
        logger.info("[PROFILEREPORT] - Calculating privacy metrics.")
        metrics["privacy"] = self.get_privacy_metrics(*dataframes)
        metrics["privacy_error_logs"] = self.get_error_logs(metrics["privacy"])
        metrics["privacy_perc_failed"] = self.get_percentage_failed(
            metrics["privacy"])

        logger.info("[PROFILEREPORT] - Calculating fidelity metrics.")
        metrics["fidelity"] = self.get_fidelity_metrics(*dataframes)
        metrics["fidelity_error_logs"] = self.get_error_logs(
            metrics["fidelity"])
        metrics["fidelity_perc_failed"] = self.get_percentage_failed(
            metrics["fidelity"])

        logger.info("[PROFILEREPORT] - Calculating utility metrics.")
        metrics["utility"] = self.get_utility_metrics(*dataframes)
        metrics["utility_error_logs"] = self.get_error_logs(metrics["utility"])
        metrics["utility_perc_failed"] = self.get_percentage_failed(
            metrics["utility"])

        return metrics
