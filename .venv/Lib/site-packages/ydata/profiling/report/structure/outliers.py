"""Defines outliers report structure."""
from typing import Dict, List, Union

from pandas import DataFrame as pdDataFrame
from pandas import set_option as pd_set_option
from ydata_profiling.config import Settings
from ydata_profiling.report.formatters import fmt, fmt_percent
from ydata_profiling.report.presentation.core import Container, Image, Sample, Table
from ydata_profiling.report.presentation.core.renderable import Renderable


def _create_pipeline_tables_rows(table: dict) -> list:
    return [
        {
            "name": "Number of clusters",
            "value": fmt(table["total_num_clusters"]),
            "alert": False,
        },
        {
            "name": "Number of outliers",
            "value": fmt(table["total_num_outliers"]),
            "alert": False,
        },
        {
            "name": "Number of outliers (%)",
            "value": fmt_percent(table["total_num_outliers_per"]),
            "alert": False,
        },
    ]


def _filter_columns(df: pdDataFrame, pipeline: str) -> pdDataFrame:
    col_to_drop = {
        "std": ["outlier_score_std", "outlier_score_min", "outlier_score_max"],
        "variance": [],
    }
    return df.drop(columns=col_to_drop[pipeline])


def _rename_columns(df: pdDataFrame) -> pdDataFrame:
    col_map = {
        "cluster_id": "Cluster",
        "num_outliers": "#Observations",
        "outlier_score_mean": "Average score",
        "outlier_score_std": "Score std",
        "outlier_score_min": "Min score",
        "outlier_score_max": "Max score",
    }
    return df.rename(columns=col_map)


def _create_pipeline_summary_tables(table: dict, name, style) -> dict:
    pd_set_option('display.precision', 4)
    table["clusters"].index = [''] * len(table["clusters"])
    table["clusters"] = _filter_columns(table["clusters"], name)
    table["clusters"] = _rename_columns(table["clusters"])

    stats = Table(
        rows=_create_pipeline_tables_rows(table),
        name="",
        style=style,
        anchor_id=f"{name}_table",
    )

    clusters = Sample(
        sample=table["clusters"],
        name="Cluster summary",
        anchor_id=f"{name}_cluster_stats",
    )
    return {"stats": stats, "clusters": clusters}


def _create_pipeline_plots(config: Settings, plots: list) -> Dict[str, Renderable]:
    if len(plots) > 0 and not isinstance(plots[0], list):
        plots = [plots]

    name_map = {
        "std": "Standard Deviation",
        "variance": "PCA projection",
    }

    n_reports = len(plots)
    n_plots = len(plots[0])

    diagrams: Dict[str, Renderable] = {}
    for plot_idx in range(n_plots):
        for report_idx in range(n_reports):
            plot = plots[report_idx][plot_idx]
            diagram: Renderable = Image(
                image=plot["image"],
                image_format=config.plot.image_format,
                alt=plot["name"],
                anchor_id=f"{plot['name']}_diagram",
                name=name_map.get(plot["name"], plot["name"]),
            )
            diagrams[plot["name"]] = diagram

    # FIXME will fail comparison
    return diagrams


def _has_outliers(node_stats: dict) -> bool:
    return node_stats["total_num_outliers"] > 0


def _create_pipeline_summary(config: Settings, summary: list) -> Renderable:
    # FIXME handle comparison
    name_map = {
        "std": "Standard Deviation",
        "variance": "Variance"
    }
    summary = summary[0]
    tabs: List[Renderable] = []
    pipeline_plots = _create_pipeline_plots(config, summary["plots"])
    for pipe in summary["summary"]["pipelines"]:
        for name, stats in pipe.items():
            if not _has_outliers(stats):
                continue
            tables = _create_pipeline_summary_tables(
                stats, name, config.html.style)
            bot = Container(
                [tables["stats"], tables["clusters"]],
                sequence_type="batch_grid",
                name="Pipeline Summary",
                batch_size=2,
                anchor_id=f"{name}_outlier_summary"
            )
            tab = Container(
                [pipeline_plots[name], bot],
                sequence_type="batch_grid",
                name=name_map.get(name, name),
                anchor_id=f"{name}_outlier_tab",
                batch_size=1,
            )
            tabs.append(tab)

    return tabs


def get_outliers_items(config: Settings, summary: Union[list, dict]) -> Renderable:
    """Create the list of outlier items.

    Args:
        config: report Settings object
        outliers: outliers data

    Returns:
        List of outliers information to show in the interface.
    """
    if isinstance(summary, dict):
        summary = [summary]

    tabs = _create_pipeline_summary(config, summary)
    tables_tab = Container(
        tabs,
        anchor_id="outliers_tabs",
        name="Outliers",
        sequence_type="tabs",
        batch_size=len(config.html.style._labels),
    )

    return tables_tab
