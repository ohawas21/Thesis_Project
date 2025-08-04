"""
    Display data quality scores overview
"""
from ydata.profiling.model import BaseDescription
from ydata_profiling.config import Settings

from ydata_profiling.report.presentation.core import Container
from ydata_profiling.report.presentation.core.scores import Scores

def get_score_color(value):
    """Function to determine color based on score thresholds."""
    if value < 50:
        return "#dc3545"  # 🔴 Red (Critical)
    elif value < 75:
        return "#ffc107"  # 🟡 Yellow (Needs Attention)
    else:
        return "#28a745"  # 🟢 Green (Good)


def get_quality_scores(config: Settings, summary: BaseDescription):
    scores = summary.scores

    if isinstance(scores['overall_score'], list):
        overall_score = [round(score*100, 2) for score in scores['overall_score']]
    else:
        overall_score = [round(scores['overall_score']*100, 2)]
    del scores['overall_score']

    scores_info = []
    for score, values in scores.items():
        if not isinstance(values, list):
            values = [values]
        submetrics = []
        for value in values:
            if value is not None:
                val = round(value*100,2)
                submetrics.append(
                    {
                        'value': val,
                        'color': get_score_color(val)
                    }
                )
        scores_info.append({'name': score.capitalize(), 'submetrics': submetrics})


    scores = [Scores(
        overall_score=overall_score,
        items=scores_info,
        name=config.html.style._labels,
        style=config.html.style
    )]

    return [Container(
        items=scores,
        sequence_type="scores",
        name="Data Quality scores",
        anchor_id="quality_scores",
    )]
