"""Generate the report."""
import os

from dataclasses import asdict
from typing import List

from dacite import from_dict
from tqdm.auto import tqdm
from ydata_profiling.config import Settings
from ydata_profiling.model.alerts import AlertType
from ydata.profiling.report.config import get_render_map
from ydata_profiling.report.presentation.core import HTML, Collapse, Container, Dropdown, ToggleButton, Variable
from ydata_profiling.report.presentation.core.renderable import Renderable
from ydata_profiling.report.presentation.core.root import Root
from ydata_profiling.report.structure.correlations import get_correlation_items
from ydata_profiling.report.structure.report import (get_duplicates_items, get_interactions, get_missing_items,
                                                     get_sample_items)


from ydata.profiling.model import BaseDescription
from ydata.profiling.report.structure.quality_overview import get_quality_scores
from ydata.profiling.report.structure.overview import get_dataset_items
from ydata.profiling.report.structure.outliers import get_outliers_items


def get_report_structure(config: Settings, summary: BaseDescription) -> Root:
    """Generate a HTML report from summary statistics and a given sample.

    Args:
      config: report Settings object
      summary: Statistics to use for the overview, variables, correlations and missing values.

    Returns:
      The profile report in HTML format
    """
    summary = from_dict(data_class=BaseDescription, data=asdict(summary))
    disable_progress_bar = not config.progress_bar
    with tqdm(
        total=1, desc="Generate report structure", disable=disable_progress_bar
    ) as pbar:
        alerts = summary.alerts

        section_items: List[Renderable] = get_quality_scores(config=config,
                                                            summary=summary)

        section_items += [
            Container(
                get_dataset_items(config, summary, alerts),
                sequence_type="overview_tabs",
                name="Overview",
                anchor_id="overview",
                oss=not bool(os.getenv("YDATA_SUPPRESS_BANNER", "")),
            ),
        ]

        if len(summary.variables) > 0:
            section_items.append(
                Dropdown(
                    name="Variables",
                    anchor_id="variables-dropdown",
                    id="variables-dropdown",
                    is_row=True,
                    classes=["dropdown-toggle"],
                    items=list(summary.variables),
                    item=Container(
                        render_variables_section(config, summary),
                        sequence_type="accordion",
                        name="Variables",
                        anchor_id="variables",
                    ),
                )
            )

        scatter_items = get_interactions(config, summary.scatter)
        if len(scatter_items) > 0:
            section_items.append(
                Container(
                    scatter_items,
                    sequence_type="tabs" if len(
                        scatter_items) <= 10 else "select",
                    name="Interactions",
                    anchor_id="interactions",
                ),
            )

        corr = get_correlation_items(config, summary)
        if corr is not None:
            section_items.append(corr)

        missing_items = get_missing_items(config, summary)
        if len(missing_items) > 0:
            section_items.append(
                Container(
                    missing_items,
                    sequence_type="tabs",
                    name="Missing values",
                    anchor_id="missing",
                )
            )

        sample_items = get_sample_items(config, summary.sample)
        if len(sample_items) > 0:
            section_items.append(
                Container(
                    items=sample_items,
                    sequence_type="tabs",
                    name="Sample",
                    anchor_id="sample",
                )
            )

        #replace this logic with the analysis from the near_duplicates
        duplicate_items = get_duplicates_items(config, summary.duplicates)
        if len(duplicate_items) > 0:
            section_items.append(
                Container(
                    items=duplicate_items,
                    sequence_type="batch_grid",
                    batch_size=len(duplicate_items),
                    name="Duplicate rows",
                    anchor_id="duplicate",
                )
            )

        #I need to create a config for the near duplicates
        near_duplicates_items = get_duplicates_items(config, summary.near_duplicates)
        if len(near_duplicates_items) > 0:
            section_items.append(
                Container(
                    items=near_duplicates_items,
                    sequence_type="batch_grid",
                    batch_size=len(near_duplicates_items),
                    name = "Near duplicate rows",
                    anchor_id= "near_duplicate",
                )
        )

        outliers = summary.outliers or {}
        if len(outliers)>0:
            section_items.append(
                get_outliers_items(config, outliers)
            )

        sections = Container(
            section_items,
            name="Root",
            sequence_type="sections",
            full_width=config.html.full_width,
        )
        pbar.update()

    footer = HTML(
        content=''
    )

    return Root("Root", sections, footer, style=config.html.style)


def render_variables_section(
    config: Settings, dataframe_summary: BaseDescription
) -> list:
    """Render the HTML for each of the variables in the DataFrame.

    Args:
        config: report Settings object
        dataframe_summary: The statistics for each variable.

    Returns:
        The rendered HTML, where each row represents a variable.
    """

    templs = []

    descriptions = config.variables.descriptions
    show_description = config.show_variable_description
    reject_variables = config.reject_variables

    render_map = get_render_map()

    for idx, summary in dataframe_summary.variables.items():
        # Common template variables
        if not isinstance(dataframe_summary.alerts, tuple):
            alerts = [
                alert.fmt()
                for alert in dataframe_summary.alerts
                if alert.column_name == idx
            ]

            alert_fields = {
                field
                for alert in dataframe_summary.alerts
                if alert.column_name == idx
                for field in alert.fields
            }

            alert_types = {
                alert.alert_type
                for alert in dataframe_summary.alerts
                if alert.column_name == idx
            }
        else:
            alerts = tuple(
                [alert.fmt()
                 for alert in summary_alerts if alert.column_name == idx]
                for summary_alerts in dataframe_summary.alerts
            )  # type: ignore

            alert_fields = {
                field
                for summary_alerts in dataframe_summary.alerts
                for alert in summary_alerts
                if alert.column_name == idx
                for field in alert.fields
            }

            alert_types = {
                alert.alert_type
                for summary_alerts in dataframe_summary.alerts
                for alert in summary_alerts
                if alert.column_name == idx
            }

        template_variables = {
            "varname": idx,
            "varid": hash(idx),
            "alerts": alerts,
            "description": descriptions.get(idx, "") if show_description else "",
            "alert_fields": alert_fields,
        }

        template_variables.update(summary)

        # Per type template variables
        if isinstance(summary["type"], list):
            types = set(summary["type"])
            if len(types) == 1:
                variable_type = list(types)[0]
            else:
                # This logic may be treated by the typeset
                if "Categorical" in types:
                    # Treating one is categorical render as categorical
                    variable_type = "Categorical"
                else:
                    variable_type = "Unsupported"

        else:
            variable_type = summary["type"]
        render_map_type = render_map.get(
            variable_type, render_map["Unsupported"])
        template_variables.update(render_map_type(config, template_variables))

        # Ignore these
        if reject_variables:
            ignore = AlertType.REJECTED in alert_types
        else:
            ignore = False

        bottom = None
        if "bottom" in template_variables and template_variables["bottom"] is not None:
            btn = ToggleButton(
                "More details", anchor_id=template_variables["varid"])
            bottom = Collapse(btn, template_variables["bottom"])

        var = Variable(
            template_variables["top"],
            bottom=bottom,
            anchor_id=template_variables["varid"],
            name=idx,
            ignore=ignore,
        )

        templs.append(var)

    return templs
