"""YData report generation class."""

from __future__ import absolute_import, division, print_function

import os
import warnings
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from typing import Optional
from warnings import warn

import dask
import pandas as pd
from numpy import clip as np_clip
from numpy import isnan as np_isnan
from numpy import mean as np_mean
from numpy import nan as np_nan
from pdf_reports import ReportWriter, preload_stylesheet, write_report
from sklearn.exceptions import ConvergenceWarning

from ydata.datascience.common import PrivacyLevel
from ydata.dataset import Dataset
from ydata.metadata import Metadata
from ydata.metadata.column import Column
from ydata.report.logger import logger
from ydata.report.metrics import MetricType
from ydata.report.metrics.score import MetricScore
from ydata.report.reports.report_status import ReportStatus
from ydata.report.reports.report_type import ReportType
from ydata.report.reports.syntheticdata.syntheticdata_metrics import SyntheticDataMetrics
from ydata.report.reports.utils import score_to_label_leq
from ydata.report.styles.html import StyleHTML
from ydata.report.styles.text import TextFormatter
from ydata.utils._ipython import isnotebook
from ydata.utils.exceptions import LessRowsThanColumns, NotEnoughRows, SmallTrainingDataset


class SyntheticDataProfile:
    """Class to generate a Synthetic Data Profile."""

    MIN_ROWS = 50
    LOW_ROWS = 100
    METRIC_ERROR_VALUE = None

    SCORE_TO_LABEL = {
        0.24: "Very Poor",
        0.49: "Poor",
        0.69: "Moderate",
        0.79: "Good",
        0.94: "Very Good",
        1.00: "Excellent"
    }

    def __init__(
        self,
        real: Dataset,
        synth: Dataset,
        metadata: Metadata,
        privacy_level: PrivacyLevel | int = PrivacyLevel.HIGH_FIDELITY,
        report_type: ReportType | str = ReportType.TABULAR,
        safe_mode: bool = True,
        data_types: dict | None = None,
        training_data: Dataset | None = None,
        target: str | Column | None = None,
        anonymized_cols: list[str] | None = None,
        synth_name: str | None = None
    ):
        """Initialization of the SyntheticDataProfile class.

        Args:
            real (Dataset): original dataset containing an holdout not used to train the synthesizer.
            synth (Dataset): synthetically generated data samples.
            metadata (Metadata): metadata of the original dataset.
            privacy_level (PrivacyLevel): the level of privacy applied to the synthetic data. Defaults to 'HIGH_FIDELITY'.
            report_type (ReportType): whether to calculate a report for 'tabular' or 'timeseries' data. Defaults to 'tabular'.
            safe_mode (bool): whether to run in safe mode or not. If True, any exception in the metrics is handled gracefully
                without crashing the report. Defaults to True.
            data_types (dict, optional): propagates the specified data_types for the calculation of the quality metrics.
            training_data (Dataset, optional): original dataset used to train the synthesizer.
                If provided, used to calculate some metric scores (e.g. membership score)
            target (str, optional): if provided, propagates the specified target for the calculation of the quality metrics.
            anonymized_cols (list[str], optional): list of columns that are anonymized.
            synth_name (str, optional): if provided, add the synthesizer name to the report.
        """
        if metadata.shape[0] <= SyntheticDataProfile.MIN_ROWS:
            raise NotEnoughRows(
                f"Not enough rows. Training dataset must have at least {SyntheticDataProfile.MIN_ROWS} rows.")

        if metadata.shape[0] < SyntheticDataProfile.LOW_ROWS:
            warn(
                f"Small training dataset detected. For optimal results, training data should have at least {SyntheticDataProfile.LOW_ROWS} rows.", SmallTrainingDataset)

        if metadata.shape[0] < metadata.shape[1]:
            warn("Training data has less rows than columns. This might lead to overfitting or degraded results.", LessRowsThanColumns)

        self.report_type = (
            report_type
            if isinstance(report_type, ReportType)
            else ReportType(report_type)
        )

        self.__map_score_to_label = lambda x: score_to_label_leq(
            x, mapping=self.SCORE_TO_LABEL)

        self.safe_mode = safe_mode
        self._report_status = None

        privacy_level = PrivacyLevel(privacy_level)
        self.anonymized_cols = [] if anonymized_cols is None else anonymized_cols
        for col in self.anonymized_cols:
            if col not in metadata.columns:
                raise ValueError(
                    "At least one of the anonymized columns does not exist in the data.")

        self.creation_date = datetime.now(timezone.utc)
        self._metrics = None

        logger.info("[PROFILEREPORT] - Starting metrics calculation.")
        if self.report_type == ReportType.TIMESERIES:
            assert isinstance(
                metadata, Metadata
            ), "Metadata is required for time series reports."

        target = self._validate_target(
            real, synth, training_data, target, metadata)
        calc_utility = True if target else False

        self._init_report_writer()

        if target:
            if isinstance(target, Column):
                target = target.name

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            self._calculate_metrics(
                real=real,
                synth=synth,
                data_types=data_types,
                training_data=training_data,
                metadata=metadata,
                target=target
            )

            info_metrics = self.__prepare_info_metrics(self._metrics)
            anonymized_metrics = self.__prepare_anonymized_metrics(
                self._metrics)
            privacy_metrics = self.__prepare_privacy_metrics(self._metrics)
            fidelity_metrics = self.__prepare_fidelity_metrics(self._metrics)
            if calc_utility:
                utility_metrics = self.__prepare_utility_metrics(self._metrics)
            else:
                utility_metrics = self.__prepare_utility_2nd_block(
                    self._metrics)

        self._update_report_status()
        self._check_report_sections()

        info_metrics["privacy_level"] = {
            "label": "Privacy Level",
            "value": str(privacy_level)
        }

        pca = self._metrics["fidelity"].get("pca_chart", {})
        created_at = f"Report Generated at {datetime.now().strftime('%m/%d/%Y, %H:%M')}"

        if pca:
            if pca.is_error:
                pca_data = {
                    "title": pca.name,
                    "dimension_reduction_description": pca.description.format("-"),
                    "error": pca.values,
                    "has_errors": True
                }
            else:
                pca_data = pca.values
            pca_data["available"] = True
        else:
            pca_data = {"available": False}

        self._report_info = {
            "synth_name": synth_name or "YData synthetic data generator",
            "created_at": created_at,
            "report_type": self.report_type.value,
            "info_metrics": info_metrics["infos"],
            "featured": info_metrics["featured"],
            "privacy_level": info_metrics["privacy_level"],
            "anonymized": anonymized_metrics,
            "privacy": privacy_metrics,
            "fidelity": fidelity_metrics,
            "utility": utility_metrics,
            "privacy_error_logs": self._metrics["privacy_error_logs"],
            "fidelity_error_logs": self._metrics["fidelity_error_logs"],
            "utility_error_logs": self._metrics["utility_error_logs"],
            "plots": pca_data
        }

        """ SCSS Preload """
        # Change the path to be an env variable with default value maybe.
        # TODO move to a method
        path_to_css = os.path.join(
            os.path.dirname(__file__), os.pardir, os.pardir)
        # TODO put paths in an object
        path_to_css = path_to_css + r"/styles/css/ydata-pdf-generator-styles.scss"
        self.css = preload_stylesheet(path_to_css)

        self.display_notebook()

    def _update_report_status(self):
        rep_status = {}
        for section in ["privacy", "fidelity", "utility"]:
            if self._metrics[f"{section}_perc_failed"] == 0.0:
                status = ReportStatus.OK
            elif self._metrics[f"{section}_perc_failed"] == 1.0:
                status = ReportStatus.FAILED
            else:
                status = ReportStatus.WARNING
            rep_status[section] = status
        self._report_status = rep_status

    def _check_report_sections(self):
        if self.report_status is None:
            self._update_report_status()

        failed_sections = [
            k for k, v in self.report_status.items() if v == ReportStatus.FAILED]
        if len(failed_sections) > 0:
            raise RuntimeError("The report cannot be generated since all metrics from " +
                               f"the following sections failed: {', '.join(failed_sections)}.")

    @property
    def report_status(self):
        return self._report_status

    def _init_report_writer(self):
        """Get the report template: time-series or regular."""
        template_name = "report.pug"

        path_to_pug = os.path.join(
            os.path.dirname(__file__), os.pardir, os.pardir)
        path_to_pug = path_to_pug + f"/styles/pug/{template_name}"

        self.report_writer = ReportWriter(
            default_template=path_to_pug,
            # TODO use TextStyle instead of <strong>
            title="Synthetic <strong>Data Quality</strong> Report",
            organization="YData",
            version="1.0.0",
        )

    def __prepare_privacy_metrics(self, metrics):
        """Prepare privacy metrics for pdf report format."""

        privacy_metric = dict()
        privacy_metric["title"] = "Privacy Scores"
        privacy_metric["title_metrics"] = "Privacy Metrics"
        privacy_metric[
            "text"
        ] = "Privacy indicates the level of confidentiality of the synthetic data. \
            Larger and more complex datasets typically offer higher privacy scores and more protection against attacks."
        privacy_metric["blocks"] = []
        privacy_metric["metrics_description"] = ""
        for score in metrics["privacy"].values():
            # Get the value and the label for each metric
            privacy_metric["blocks"].append(
                {"value": round(
                    score.values, 2) if not score.is_error else self.METRIC_ERROR_VALUE, "label": score.name}
            )
            # Add the description to the metrics description
            privacy_metric["metrics_description"] += StyleHTML.paragraph(
                score.description)
        privacy_metric["blocks_available"] = any(
            [b["value"] != self.METRIC_ERROR_VALUE for b in privacy_metric["blocks"]])
        privacy_metric["has_errors"] = any(
            [s.is_error for s in metrics["privacy"].values()])
        return privacy_metric

    def __prepare_utility_metrics(self, metrics):
        """Prepare utility metrics for pdf report format."""

        utility_metric = dict()
        utility_metric["title"] = "Utility Scores"
        utility_metric[
            "text"
        ] = "YData's Synthetic Data Profile utility scores are grouped into two main classes:<br/> \
            &#8226 Query quality enables you to understand the quality of the generated synthetic data to answer the \
            questions with similar results as the original data.<br/> \
            &#8226 The Predictive performance allows you to understand the effects of an ML model trained on your \
            synthetic data and later applied in production applications. It provides a supervised perspective \
            of the synthetic data generated.<br/> \
            The anonymized columns are not considered for the utility scores since they are necessarily different \
            between the real and synthetic records in order to achieve anonymization."
        utility_metric["tstr_title"] = "Prediction Metrics"

        tstr_data = metrics["utility"]["tstr"]

        # TODO This should be changed into enums. This mapping wouldn't be required with enum
        METRIC_NAME_MAP = defaultdict(lambda: "performance metric")
        METRIC_NAME_MAP["mae"] = ("mean absolute error",)
        METRIC_NAME_MAP["accuracy"] = ("accuracy",)
        METRIC_NAME_MAP["auc"] = ("ROC-AUC",)
        METRIC_NAME_MAP["f1"] = "F1-Score"
        METRIC_NAME_MAP["mape"] = "mean absolute percentage error"

        utility_metric["block"] = {
            "tstr": TextFormatter.format_float(tstr_data.values["score_tstr"])
            if not tstr_data.is_error else self.METRIC_ERROR_VALUE,
            "tstr_label": "tstr",
            "trtr": TextFormatter.format_float(tstr_data.values["score_trtr"])
            if not tstr_data.is_error else self.METRIC_ERROR_VALUE,
            "trtr_label": "trtr",
            "trtr_ratio": TextFormatter.format_float(
                tstr_data.values["score_tstr"] / tstr_data.values["score_trtr"]
            ) if not tstr_data.is_error else self.METRIC_ERROR_VALUE,
            "trtr_ratio_label": "score",
            "task": tstr_data.values["task"] if not tstr_data.is_error else self.METRIC_ERROR_VALUE,
            "target variable": tstr_data.values["target_variable"]
            if not tstr_data.is_error else self.METRIC_ERROR_VALUE,
            "metric": tstr_data.values["metric"] if not tstr_data.is_error else self.METRIC_ERROR_VALUE,
            "description": "TSTR",
        }
        utility_metric["tstr_dataframe"] = pd.DataFrame(
            data=tstr_data.values["detailed"], columns=tstr_data.values["detailed"].keys(
            )
        ) if not tstr_data.is_error else pd.DataFrame()
        utility_metric["metrics_description_1"] = StyleHTML.paragraph(
            metrics["utility"]["tstr"].description)
        block2 = self.__prepare_utility_2nd_block(metrics)
        for k, v in block2.items():
            if k not in utility_metric:
                utility_metric[k] = v

        if self.report_type == ReportType.TABULAR:
            feat = {}
            feat["title"] = "Feature Importance"
            feat["desc"] = metrics["utility"]["feature_importance"].description
            feat["plot"] = metrics["utility"]["feature_importance"].values["plot"] \
                if not metrics["utility"]["feature_importance"].is_error else None
            feat["score"] = TextFormatter.format_float(
                metrics["utility"]["feature_importance"].values["score"]) \
                if not metrics["utility"]["feature_importance"].is_error else self.METRIC_ERROR_VALUE
            feat["is_error"] = metrics["utility"]["feature_importance"].is_error
            feat["name"] = metrics["utility"]["feature_importance"].name
            utility_metric["feature_importance"] = feat

        return utility_metric

    def __prepare_utility_2nd_block(self, metrics):
        utility_metric = dict()
        utility_metric["title"] = "Utility Scores"
        utility_metric[
            "text"
        ] = "YData's Synthetic Data Profile utility scores are grouped into two main classes:<br/> \
            &#8226 Query quality enables you to understand the quality of the generated synthetic data to answer the \
            questions with similar results as the original data.<br/> \
            &#8226 The Predictive performance allows you to understand the effects of an ML model trained on your \
            synthetic data and later applied in production applications. It provides a supervised perspective \
            of the synthetic data generated.<br/> \
            The anonymized columns are not considered for the utility scores since they are necessarily different \
            between the real and synthetic records in order to achieve anonymization."
        utility_metric["qscore_title"] = "QScore"
        b2_metrics = ["qscore"]
        utility_metric["metrics_description_2"] = ""
        utility_metric["block2"] = []
        for k, score in metrics["utility"].items():
            if k in b2_metrics:
                utility_metric["block2"].append(
                    {
                        "value": TextFormatter.format_float(score.values)
                        if not score.is_error else self.METRIC_ERROR_VALUE,
                        "label": score.name,
                    }
                )
                utility_metric["metrics_description_2"] += StyleHTML.paragraph(
                    score.description)
        utility_metric["blocks2_available"] = any(
            [b["value"] != self.METRIC_ERROR_VALUE for b in utility_metric["block2"]])
        utility_metric["has_errors"] = any(
            [s.is_error for k, s in metrics["utility"].items() if k in b2_metrics])

        return utility_metric

    def __prepare_fidelity_metrics(self, metrics):
        """Prepare fidelity metrics for pdf report format."""
        fidelity_metric = dict()
        fidelity_metric["title"] = "Fidelity Scores"
        fidelity_metric[
            "text"
        ] = "Fidelity measures how well the synthetic data statistically \
            matches the original records. It is provided through univariate and multivariate metrics, \
            model and assumption-free. The anonymized columns are not considered for the fidelity scores \
            since they are necessarily different between the real and synthetic records in order to \
            achieve anonymization."
        fidelity_metric["stats_title"] = "Statistics Distance"
        fidelity_metric[
            "stats_text"
        ] = "The metrics below provide summarized information of the detailed \
            statistical metrics calculated through YData's profiling."
        fidelity_metric["metrics_description"] = ""

        fidelity_metric["missing_values_similarity"] = {
            "title": metrics["fidelity"]["missing_values_similarity"].name,
            "description": metrics["fidelity"]["missing_values_similarity"].description,
            "mean": TextFormatter.format_float(metrics["fidelity"]["missing_values_similarity"].values["mean"])
            if not metrics["fidelity"]["missing_values_similarity"].is_error else self.METRIC_ERROR_VALUE,
            "dataframes": {
                "mvs_df_top_k_cols": metrics["fidelity"]["missing_values_similarity"].values["dataframes"]["table_top_k_cols"]
                if not metrics["fidelity"]["missing_values_similarity"].is_error else pd.DataFrame(),
                "mvs_df_top_bottom_cols": metrics["fidelity"]["missing_values_similarity"].values["dataframes"]["table_bottom_k_cols"]
                if not metrics["fidelity"]["missing_values_similarity"].is_error else pd.DataFrame(),
            },
        }

        mean_blocks = []
        for block_keys in [["Mean", "Std. Dev."], ["Median", "Q25%", "Q75%"]]:
            mean_blocks.append([
                {
                    "label": key,
                    "value": TextFormatter.format_float(
                        metrics["fidelity"]["statistical_similarity"].values["mean"][key])
                    if not metrics["fidelity"]["statistical_similarity"].is_error else self.METRIC_ERROR_VALUE
                }
                for key in block_keys
            ])

        fidelity_metric["statistical_similarity"] = {
            "title": metrics["fidelity"]["statistical_similarity"].name,
            "description": metrics["fidelity"]["statistical_similarity"].description,
            "mean_blocks": mean_blocks,
            "has_errors": any([b["value"] == self.METRIC_ERROR_VALUE for b in
                               [block for block_group in mean_blocks for block in block_group]]),
            "dataframe": metrics["fidelity"]["statistical_similarity"].values["dataframe"]
            if not metrics["fidelity"]["statistical_similarity"].is_error else pd.DataFrame()
        }

        fidelity_metric["distribution"] = {
            "title": "Distribution Metrics",
            "text": "The distribution metrics compare the probability distributions of the \
                real and synthetic data variables. These metrics return values between [0, 1],\
                where closer to 1 is desirable (i.e., the distributions are likely the same).",
            "mean": {
                "ks_test": {
                    "label": "Kolmogorov-Smirnov Test",
                    "value": TextFormatter.format_float(metrics["fidelity"]["ks_test"].values["mean"])
                    if not metrics["fidelity"]["ks_test"].is_error else self.METRIC_ERROR_VALUE
                },
                "total_variation_distance": {
                    "label": "Total Variation Distance",
                    "value": TextFormatter.format_float(metrics["fidelity"]["total_variation_distance"].values["mean"])
                    if not metrics["fidelity"]["total_variation_distance"].is_error else self.METRIC_ERROR_VALUE
                },
            },
            "dataframes": {
                "ks_df_top_k_cols": metrics["fidelity"]["ks_test"].values["dataframes"]["table_top_k_cols"]
                if not metrics["fidelity"]["ks_test"].is_error else pd.DataFrame(),
                "ks_df_top_bottom_cols": metrics["fidelity"]["ks_test"].values["dataframes"]["table_bottom_k_cols"]
                if not metrics["fidelity"]["ks_test"].is_error else pd.DataFrame(),
                "total_variation_df_top_k_cols": metrics["fidelity"]["total_variation_distance"].values["dataframes"]["table_top_k_cols"]
                if not metrics["fidelity"]["total_variation_distance"].is_error else pd.DataFrame(),
                "total_variation_df_top_bottom_cols": metrics["fidelity"]["total_variation_distance"].values["dataframes"]["table_bottom_k_cols"]
                if not metrics["fidelity"]["total_variation_distance"].is_error else pd.DataFrame()
            },
            "description": {
                "ks_test": metrics["fidelity"]["ks_test"].description,
                "total_variation_distance": metrics["fidelity"]["total_variation_distance"].description
            }
        }

        fidelity_metric["coverage"] = {
            "title": "Coverage Metrics",
            "text": "The coverage metrics describe how well the real data variables are represented \
                in the synthetic data and they return values between [0, 1], where 1 represents complete \
                coverage. These metrics are divided into two groups:<br>&#8226 Metrics for categorical data, \
                which include <b>Category Coverage</b> and <b>Missing Category Coverage</b>.<br>&#8226 Metrics \
                for numerical data, namely the <b>Range Coverage</b>.",
            "title_cat": "Categorical Data",
            "text_cat": "The following metrics are specific to categorical data.",
            "title_num": "Numerical Data",
            "text_num": "The following metrics are specific to numerical data.",
            "mean": {
                "cc": {
                    "label": "Category Coverage",
                    "value": TextFormatter.format_float(metrics["fidelity"]["category_coverage"].values["mean"])
                        if not metrics["fidelity"]["category_coverage"].is_error else self.METRIC_ERROR_VALUE
                },
                "mcc": {
                    "label": "Missing Category Coverage",
                    "value": TextFormatter.format_float(metrics["fidelity"]["missing_category_coverage"].values["mean"])
                    if not metrics["fidelity"]["missing_category_coverage"].is_error else self.METRIC_ERROR_VALUE
                },
                "rc": {
                    "label": "Range Coverage",
                    "value": TextFormatter.format_float(metrics["fidelity"]["range_coverage"].values["mean"])
                    if not metrics["fidelity"]["range_coverage"].is_error else self.METRIC_ERROR_VALUE
                },
            },
            "dataframes": {
                "cc_df_top_k_cols": metrics["fidelity"]["category_coverage"].values["dataframes"]["table_top_k_cols"]
                if not metrics["fidelity"]["category_coverage"].is_error else pd.DataFrame(),
                "cc_df_top_bottom_cols": metrics["fidelity"]["category_coverage"].values["dataframes"]["table_bottom_k_cols"]
                if not metrics["fidelity"]["category_coverage"].is_error else pd.DataFrame(),
                "mcc_df_top_k_cols": metrics["fidelity"]["missing_category_coverage"].values["dataframes"]["table_top_k_cols"]
                if not metrics["fidelity"]["missing_category_coverage"].is_error else pd.DataFrame(),
                "mcc_df_top_bottom_cols": metrics["fidelity"]["missing_category_coverage"].values["dataframes"]["table_bottom_k_cols"]
                if not metrics["fidelity"]["missing_category_coverage"].is_error else pd.DataFrame(),
                "rc_df_top_k_cols": metrics["fidelity"]["range_coverage"].values["dataframes"]["table_top_k_cols"]
                if not metrics["fidelity"]["range_coverage"].is_error else pd.DataFrame(),
                "rc_df_top_bottom_cols": metrics["fidelity"]["range_coverage"].values["dataframes"]["table_bottom_k_cols"]
                if not metrics["fidelity"]["range_coverage"].is_error else pd.DataFrame()
            },
            "description": {
                "cc": metrics["fidelity"]["category_coverage"].description,
                "mcc": metrics["fidelity"]["missing_category_coverage"].description,
                "rc": metrics["fidelity"]["range_coverage"].description
            }
        }

        block_metrics = [
            "distance_correlation",
            "mean_ratio",
            "std_ratio",
            "distance_distribution",
            "autocorrelation",
            "synth_classifier"
        ]

        if self.report_type == ReportType.TABULAR:
            block_list = []
            for k, score in metrics["fidelity"].items():
                if k in block_metrics:
                    block_list.append(
                        {
                            "value": TextFormatter.format_float(score.values)
                            if not score.is_error else self.METRIC_ERROR_VALUE,
                            "label": score.name,
                        }
                    )
                    fidelity_metric["metrics_description"] += StyleHTML.paragraph(
                        score.description)
            fidelity_metric["blocks"] = [block_list]
            fidelity_metric["blocks_available"] = \
                any([b["value"] != self.METRIC_ERROR_VALUE for b in block_list])

        elif self.report_type == ReportType.TIMESERIES:
            block1, block2 = [], []
            block1_metrics = [
                "distance_correlation",
                "distance_distribution",
                "synth_classifier"
            ]
            for k, score in metrics["fidelity"].items():
                if k == "autocorrelation":
                    block2.append({
                        "value": TextFormatter.format_float(score.values["real_score"])
                        if not score.is_error else self.METRIC_ERROR_VALUE,
                        "label": f"{score.name} Real",
                        "_label": score.name,
                    })
                    block2.append({
                        "value": TextFormatter.format_float(score.values["synth_score"])
                        if not score.is_error else self.METRIC_ERROR_VALUE,
                        "label": f"{score.name} Synth",
                    })
                    fidelity_metric["metrics_description"] += StyleHTML.paragraph(
                        score.description)

                elif k in block_metrics:
                    info = {
                        "value": TextFormatter.format_float(score.values)
                        if not score.is_error else self.METRIC_ERROR_VALUE,
                        "label": score.name,
                    }
                    if k in block1_metrics:
                        block1.append(info)
                    else:
                        block2.append(info)
                    fidelity_metric["metrics_description"] += StyleHTML.paragraph(
                        score.description)

            fidelity_metric["blocks"] = [block1, block2]
            fidelity_metric["blocks_available"] = \
                any([b["value"] != self.METRIC_ERROR_VALUE for b in block1]) or \
                any([b["value"] != self.METRIC_ERROR_VALUE for b in block2])

        fidelity_metric["has_errors"] = any(
            [s.is_error for k, s in metrics["fidelity"].items() if k in block_metrics])
        for k, score in metrics["fidelity"].items():
            if score.type == MetricType.VISUAL:
                if score.is_error:
                    fidelity_metric[k] = {
                        "title": score.name,
                        "description": score.description,
                        "error": score.values,
                        "has_errors": True,
                        "mean_str": self.METRIC_ERROR_VALUE
                    }
                else:
                    fidelity_metric[k] = score.values
                    fidelity_metric[k]["has_errors"] = False
                    if k == "mutual_information":
                        fidelity_metric[k]["mean_str"] = \
                            TextFormatter.format_float(
                                fidelity_metric[k]["mean"])

        return fidelity_metric

    def __get_featured_utility(self, metrics: dict):
        utility = []
        if not metrics["utility"]["qscore"].is_error:
            utility.append(metrics["utility"]["qscore"].values)
        if "feature_importance" in metrics["utility"] and not metrics["utility"]["feature_importance"].is_error:
            utility.append(metrics["utility"]
                           ["feature_importance"].values["score"])
        if "tstr" in metrics["utility"] and not metrics["utility"]["tstr"].is_error:
            tstr = metrics["utility"]["tstr"]
            if tstr.values["task"] == "classification":
                utility.append(
                    tstr.values["score_tstr"] / tstr.values["score_trtr"])
            else:
                utility.append(
                    tstr.values["score_trtr"] / tstr.values["score_tstr"])

        return np_clip(np_mean(utility), 0, 1) if len(utility) > 0 else np_nan

    def __get_featured_fidelity(self, metrics: dict):
        metrics = metrics["fidelity"]
        fidelity = []
        if not metrics["distance_distribution"].is_error:
            fidelity.append(metrics["distance_distribution"].values)
        if not metrics["missing_values_similarity"].is_error:
            fidelity.append(
                metrics["missing_values_similarity"].values["mean"])
        if not metrics["mutual_information"].is_error:
            fidelity.append(metrics["mutual_information"].values["mean"])
        if not metrics["synth_classifier"].is_error:
            fidelity.append(metrics["synth_classifier"].values)

        if not metrics["distance_correlation"].is_error and not np_isnan(metrics["distance_correlation"].values):
            fidelity.append(metrics["distance_correlation"].values)
        if not metrics["statistical_similarity"].is_error and not np_isnan(list(metrics["statistical_similarity"].values["mean"].values())).any():
            for metric_mean in list(metrics["statistical_similarity"].values["mean"].values()):
                fidelity.append(metric_mean)
        if not metrics["category_coverage"].is_error and not np_isnan(metrics["category_coverage"].values["mean"]):
            fidelity.append(metrics
                            ["category_coverage"].values["mean"])
        if not metrics["missing_category_coverage"].is_error and not np_isnan(metrics["missing_category_coverage"].values["mean"]):
            fidelity.append(
                metrics["missing_category_coverage"].values["mean"])
        if not metrics["range_coverage"].is_error and not np_isnan(metrics["range_coverage"].values["mean"]):
            fidelity.append(metrics
                            ["range_coverage"].values["mean"])
        if not metrics["ks_test"].is_error and not np_isnan(metrics["ks_test"].values["mean"]):
            fidelity.append(metrics["ks_test"].values["mean"])
        if not metrics["total_variation_distance"].is_error and not np_isnan(metrics["total_variation_distance"].values["mean"]):
            fidelity.append(metrics
                            ["total_variation_distance"].values["mean"])

        return np_clip(np_mean(fidelity), 0, 1) if len(fidelity) > 0 else np_nan

    def __get_featured_privacy(self, metrics: Optional[dict] = None):
        privacy = []
        metrics_list = [
            "exact_matches",
            "identifiability",
            "membership",
        ]
        privacy += [
            1 - metrics["privacy"][m].values
            for m in metrics_list
            if m in metrics["privacy"] and not metrics["privacy"][m].is_error
        ]

        return np_clip(np_mean(privacy), 0, 1) if len(privacy) > 0 else np_nan

    def __get_featured_scores(self, metrics: dict | None = None):
        "Returns the utility, privacy and fidelity scores to be featured in the report landing page."

        if metrics is None:
            return "N/A", "N/A", "N/A"

        summary = metrics["summary"]
        fidelity = summary["fidelity"]
        fidelity_featured = TextFormatter.float_to_pct(fidelity, decimals=0) \
            if not np_isnan(fidelity) else "N/A"
        utility = summary["utility"]
        utility_featured = TextFormatter.float_to_pct(utility, decimals=0) \
            if not np_isnan(utility) else "N/A"
        privacy = summary["privacy"]
        privacy_featured = TextFormatter.float_to_pct(privacy, decimals=0) \
            if not np_isnan(privacy) else "N/A"

        return [
            {
                "label": "Fidelity Score",
                "rotation": round(fidelity * 180)
                if not np_isnan(fidelity) else 0,
                "value": fidelity_featured,
                "qualitative_label": summary["fidelity_label"]
                if not np_isnan(fidelity) else "N/A"
            },
            {
                "label": "Utility Score",
                "rotation": round(utility * 180)
                if not np_isnan(utility) else 0,
                "value": utility_featured,
                "qualitative_label": summary["utility_label"]
                if not np_isnan(utility) else "N/A"
            },
            {
                "label": "Privacy Score",
                "rotation": round(privacy * 180)
                if not np_isnan(privacy) else 0,
                "value": privacy_featured,
                "qualitative_label": summary["privacy_label"]
                if not np_isnan(privacy) else "N/A"
            },
        ]

    def __prepare_info_metrics(self, metrics):
        """Prepare info metrics for pdf report format."""
        info = metrics["info"]

        res = dict()
        res["featured"] = self.__get_featured_scores(metrics)
        res["infos"] = [
            {
                "label": "# Real data records",
                "value": TextFormatter.int_to_string(info["nrows_real"]),
            },
            {
                "label": "# Synthetic data records generated",
                "value": TextFormatter.int_to_string(info["nrows_synth"]),
            },
            {"label": "# Columns", "value": info["columns_details"]},
        ]

        return res

    def __prepare_anonymized_metrics(self, metrics):
        """Prepare anonymized metrics for pdf report format."""
        anonymized = metrics["anonymized"]
        res = dict()
        res["title"] = "Anonymized Features"
        res["description"] = "Anonymized features are columns that have been processed by YData's \
            synthesizers in order to remove their original characteristics. As a consequence, the \
            synthetic data of such features cannot be traced back to the real data (i.e., no \
            re-identification risk). This type of approach can be used with any column but is \
            particularly helpful to remove Personal Identifiable Information (PII) from a synthetic \
            dataset. Anonymized features are only considered for the privacy metrics, and their \
            characteristics are presented in this section."
        res["n_anonymized_cols_label"] = "# Anonymized Columns"
        res["n_anonymized_cols"] = anonymized["n_anonymized_cols"]
        res["anonymized_cols_details"] = anonymized["anonymized_cols_details"]
        return res

    def __pdf_generation(self, report_info, output_path):
        """Auxiliary method to generate the pdf levering the lib pdf_generator.

        Args:
             report_info (dict): A dictionary containing all the calculated info to be displayed in the html
             output_path (str): The path where the .pdf is to be written and stored
        """

        html = self.report_writer.pug_to_html(
            infos=report_info["info_metrics"],
            featured=report_info["featured"],
            privacy_level=report_info["privacy_level"],
            anonymized=report_info["anonymized"],
            privacy=report_info["privacy"],
            fidelity=report_info["fidelity"],
            utility=report_info["utility"],
            privacy_error_logs=report_info["privacy_error_logs"],
            fidelity_error_logs=report_info["fidelity_error_logs"],
            utility_error_logs=report_info["utility_error_logs"],
            plots=report_info["plots"],
            synth_name=report_info["synth_name"],
            created_at=report_info["created_at"],
        )
        """ HTML To PDF Convertion """
        import logging

        for log in logging.root.manager.loggerDict:
            if log.startswith("fontTools") or log.startswith("weasyprint"):
                logging.getLogger(log).setLevel(logging.ERROR)

        write_report(html, output_path, extra_stylesheets=[self.css])

    def _validate_target(
        self,
        real: Dataset,
        synth: Dataset,
        training_data: Dataset | None = None,
        target: str | Column | None = None,
        metadata: Metadata | None = None,
    ):
        if target is None:
            return target

        msg = f"Specified target {target} is not available in " + "{} dataset"
        assert target in real, msg.format("real")
        assert target in synth, msg.format("synthetic")
        if training_data is not None:
            assert target in training_data, msg.format("training")

        msg = (
            f"[PROFILEREPORT] - The specified target {target} "
            + "variable is not valid. The column provided {}. Skipping some metrics."
        )

        if metadata:
            targetable_columns, details = metadata.get_possible_targets()
            if target not in targetable_columns:
                logger.info(msg.format(details[target]))
                return None

        target_info = dask.compute(
            {
                "real_missing": real._data[target].isnull().sum(),
                "real_unique": real._data[target].nunique(),
                "synth_unique": synth._data[target].nunique(),
            }
        )[0]

        real_missing = target_info["real_missing"]
        real_nunique_labels = target_info["real_unique"]
        synth_nunique_labels = target_info["synth_unique"]

        if real_missing > 0:
            logger.warning(msg.format("has missing values"))
            return None

        if real_nunique_labels == 1 or synth_nunique_labels == 1:
            logger.warning(msg.format("is constant"))
            return None

        if real_nunique_labels == len(real) or synth_nunique_labels == len(synth):
            logger.warning(msg.format("is ID-like"))
            return None

        return target

    def __convert_metric_scores_to_values(self, metrics: dict):
        metrics_cp = deepcopy(metrics)
        for k, d in metrics.items():
            if k == "pca_chart":
                del metrics_cp[k]
            elif isinstance(d, dict):
                metrics_cp[k] = self.__convert_metric_scores_to_values(
                    metrics_cp[k])
            elif isinstance(d, MetricScore):
                if d.is_error:
                    metrics_cp[k] = np_nan
                else:
                    if isinstance(d.values, dict) and "mean" in d.values:
                        metrics_cp[k] = d.values["mean"]
                    elif k == "feature_importance":
                        metrics_cp[k] = d.values["score"]
                    else:
                        metrics_cp[k] = d.values
        return metrics_cp

    def _calculate_metrics(
        self,
        real: Dataset,
        synth: Dataset,
        metadata: Metadata,
        data_types: dict | None = None,
        training_data: Dataset | None = None,
        target: str | Column | None = None
    ):
        """Calculate the synthetic quality metrics and generate the plots for
        the report.

        Args:
            real (Dataset): original dataset containing an holdout not used to train the synthesizer.
            synth (Dataset): synthetically generated data samples.
            metadata (Metadata): metadata of the original dataset.
            data_types (dict, optional): propagates the specified data_types for the calculation of the quality metrics.
            training_data (Dataset, optional): original dataset used to train the synthesizer.
                If provided, used to calculate some metric scores (e.g. membership score)
            target (str, optional): if provided, propagates the specified target for the calculation of the quality metrics.
        """
        if data_types is None:
            data_types = {k: v.datatype for k, v in metadata.columns.items()}

        evaluator = SyntheticDataMetrics(
            real=real,
            synth=synth,
            target=target,
            safe_mode=self.safe_mode,
            anonymized_cols=self.anonymized_cols,
            training_data=training_data,
            data_types=data_types,
            report_type=self.report_type,
            metadata=metadata
        )
        self._metrics = evaluator.evaluate()
        fidelity = self.__get_featured_fidelity(self._metrics)
        privacy = self.__get_featured_privacy(self._metrics)
        utility = self.__get_featured_utility(self._metrics)
        self._metrics["summary"] = {
            "fidelity": fidelity,
            "fidelity_label": self.__map_score_to_label(fidelity)
            if not np_isnan(fidelity) else "N/A",
            "privacy": privacy,
            "privacy_label": self.__map_score_to_label(privacy)
            if not np_isnan(privacy) else "N/A",
            "utility": utility,
            "utility_label": self.__map_score_to_label(utility)
            if not np_isnan(utility) else "N/A"
        }

    def get_all_metrics(self):
        """Returns all types of metrics."""
        if self._metrics is None:
            return {}
        all_metrics = self.__convert_metric_scores_to_values(self._metrics)
        del all_metrics["privacy_error_logs"]
        del all_metrics["privacy_perc_failed"]
        del all_metrics["fidelity_error_logs"]
        del all_metrics["fidelity_perc_failed"]
        del all_metrics["utility_error_logs"]
        del all_metrics["utility_perc_failed"]
        return all_metrics

    def get_summary(self):
        """Returns the summary metrics."""
        if self._metrics is None:
            return {}
        return self._metrics["summary"]

    def get_fidelity_metrics(self):
        """Returns the fidelity metrics."""
        if self._metrics is None:
            return {}
        return self.__convert_metric_scores_to_values(self._metrics["fidelity"])

    def get_utility_metrics(self):
        """Returns the utility metrics."""
        if self._metrics is None:
            return {}
        return self.__convert_metric_scores_to_values(self._metrics["utility"])

    def get_privacy_metrics(self):
        """Returns the privacy metrics."""
        if self._metrics is None:
            return {}
        return self.__convert_metric_scores_to_values(self._metrics["privacy"])

    def generate_report(self, output_path="./ydata-report.pdf"):
        """Generates a .pdf report with synthetic data quality metrics.

        Args:
            output_path (str, optional): output path for the .pdf file.
        """
        if self._metrics is None:
            raise RuntimeError("The metrics were not yet calculated.")

        self.__pdf_generation(self._report_info, output_path)

    def display_notebook(self):
        if isnotebook():
            from IPython.display import HTML, display

            Title = "<h1>Metrics report</h1>"
            metrics_info = "<p><h2>Quality & Privacy summary</h2></p>"

            display(HTML(Title))
            display(HTML(metrics_info))
            for element in self._report_info["info_metrics"]:
                print("\033[1m{}\033[0m: {}".format(
                    element["label"], element["value"]))

            for element in self._report_info["featured"]:
                print("\033[1m{}\033[0m: {}".format(
                    element["label"], element["qualitative_label"]))

            display(HTML("<p><h3>Fidelity</h3></p>"))
            for block in self._report_info["fidelity"]["blocks"]:
                if isinstance(block, list):
                    for element in block:
                        print(
                            "\033[1m{}\033[0m: {}".format(
                                element["label"], element["value"]
                            )
                        )
                else:
                    print("\033[1m{}\033[0m: {}".format(
                        block["label"], block["value"]))

            if self._report_info["fidelity"]["mutual_information"] is not None:
                print("\033[1m{}\033[0m: {}".format(
                    self._report_info["fidelity"]["mutual_information"]["title"],
                    self._report_info["fidelity"]["mutual_information"]["mean_str"]))

            print("\033[1m{}\033[0m: {}".format(
                self._report_info["fidelity"]["missing_values_similarity"]["title"],
                self._report_info["fidelity"]["missing_values_similarity"]["mean"]))

            for block_row in self._report_info["fidelity"]["statistical_similarity"]["mean_blocks"]:
                for score in block_row:
                    print("\033[1m{}\033[0m: {}".format(
                        f"{score['label']} Similarity", score["value"]))

            print("\033[1m{}\033[0m: {}".format(
                self._report_info["fidelity"]["distribution"]["mean"]["ks_test"]["label"],
                self._report_info["fidelity"]["distribution"]["mean"]["ks_test"]["value"]))

            print("\033[1m{}\033[0m: {}".format(
                self._report_info["fidelity"]["distribution"]["mean"]["total_variation_distance"]["label"],
                self._report_info["fidelity"]["distribution"]["mean"]["total_variation_distance"]["value"]))

            print("\033[1m{}\033[0m: {}".format(
                self._report_info["fidelity"]["coverage"]["mean"]["cc"]["label"],
                self._report_info["fidelity"]["coverage"]["mean"]["cc"]["value"]))

            print("\033[1m{}\033[0m: {}".format(
                self._report_info["fidelity"]["coverage"]["mean"]["mcc"]["label"],
                self._report_info["fidelity"]["coverage"]["mean"]["mcc"]["value"]))

            print("\033[1m{}\033[0m: {}".format(
                self._report_info["fidelity"]["coverage"]["mean"]["rc"]["label"],
                self._report_info["fidelity"]["coverage"]["mean"]["rc"]["value"]))

            # TODO revisit the block logic for the utility, to align with Fidelity and Privacy
            display(HTML("<p><h3>Utility</h3></p>"))
            qscore = self._report_info["utility"]["block2"]
            for element in qscore:
                print("\033[1m{}\033[0m: {}".format(
                    element["label"], element["value"]))

            feature_importance = self._report_info["utility"].get(
                "feature_importance", None)
            if feature_importance is not None:
                print(
                    "\033[1m{}\033[0m: {}".format(
                        feature_importance["name"], feature_importance["score"]
                    )
                )

            display(HTML("<p><h3>Privacy</h3></p>"))
            for element in self._report_info["privacy"]["blocks"]:
                print("\033[1m{}\033[0m: {}".format(
                    element["label"], element["value"]))
