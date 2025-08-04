import warnings
from dataclasses import asdict
from typing import List, Optional

from dacite import from_dict
from ydata_profiling.compare_reports import (_apply_config, _compare_profile_report_preprocess, _compare_title,
                                             _create_placehoder_alerts, _placeholders, _update_merge, _update_titles)
from ydata_profiling.config import Settings

from ydata.profiling.model import BaseDescription
from ydata.profiling.profile_report import ProfileReport


def validate_reports(
    reports: List[ProfileReport], configs: List[Settings]
) -> None:
    """Validate if the reports are comparable.

    Args:
        reports: two reports to compare
                 input may either be a ProfileReport, or the summary obtained from report.get_description()
    """
    if len(reports) < 2:
        raise ValueError(
            "At least two reports are required for this comparison")

    if len(reports) > 2:
        warnings.warn(
            "Comparison of more than two reports is not supported. "
            "Reports may be produced, but may yield unexpected formatting."
        )

    report_types = [c.vars.timeseries.active for c in configs]
    if all(report_types) != any(report_types):
        raise ValueError(
            "Comparison between timeseries and tabular reports is not supported."
        )

    is_df_available = [r.df is not None for r in reports[1:]]
    if not all(is_df_available):
        raise ValueError("Reports where not initialized with a DataFrame.")
    features = [
        set(r.df.columns)
        if r.df is not None
        else set(r.description_set.variables.keys())
        for r in reports
    ]

    if not all(features[0] == x for x in features):
        warnings.warn(
            "The datasets being profiled have a different set of columns. "
            "Only the left side profile will be calculated."
        )
    if not all(isinstance(report, ProfileReport) for report in reports):
        raise TypeError(
            "The input must have the same data type for all reports. Comparing ProfileReport objects to summaries obtained from the get_description() method is not supported."
        )


def compare(
    reports: List[ProfileReport],
    config: Optional[Settings] = None,
    compute: bool = False,
) -> ProfileReport:
    """Compare Profile reports.

    Args:
        reports: two reports to compare
                 input may either be a ProfileReport, or the summary obtained from report.get_description()
        config: the settings object for the merged ProfileReport
        compute: recompute the profile report using config or the left report config
                 recommended in cases where the reports were created using different settings
    """
    if len(reports) == 0:
        raise ValueError("No reports available for comparison.")

    report_dtypes = [type(r) for r in reports]
    if len(set(report_dtypes)) > 1:
        raise TypeError(
            "The input must have the same data type for all reports. Comparing ProfileReport objects to summaries obtained from the get_description() method is not supported."
        )

    all_configs = [r.config for r in reports]
    validate_reports(reports=reports, configs=all_configs)

    base_features = list(reports[0].description_set.variables.keys())
    for report in reports[1:]:
        cols_2_compare = [
            col
            for col in base_features
            if col in report.df.columns
        ]
        report.df = report.df.loc[:, cols_2_compare]
    reports = [r for r in reports if r.description_set or not r.df.empty]
    if len(reports) == 1:
        return reports[0]

    _config = None
    if config is None:
        _config = all_configs[0].copy()
    else:
        _config = config.copy()
        for report in reports:
            tsmode = report.config.vars.timeseries.active
            title = report.config.title
            report.config = config.copy()
            report.config.title = title
            report.config.vars.timeseries.active = tsmode
            if compute:
                report._description_set = None

    _update_titles(reports)
    labels, descriptions = _compare_profile_report_preprocess(
        reports, _config)

    _config.html.style._labels = labels
    _placeholders(descriptions)

    descriptions_dict = [
        asdict(_apply_config(d, _config))
        for d in descriptions
    ]

    res: dict = _update_merge(None, descriptions_dict[0])
    for r in descriptions_dict[1:]:
        res = _update_merge(res, r)

    res["analysis"]["title"] = _compare_title(res["analysis"]["title"])
    res["alerts"] = _create_placehoder_alerts(res["alerts"])
    if not any(res["time_index_analysis"]):
        res["time_index_analysis"] = None
    profile = ProfileReport(None, config=_config)
    profile._description_set = from_dict(data_class=BaseDescription, data=res)
    # FIXME oss config not copying the private vars
    profile.config.html.style._labels = _config.html.style._labels
    return profile
