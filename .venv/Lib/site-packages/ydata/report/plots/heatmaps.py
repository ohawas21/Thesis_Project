"""Heatmaps."""
from __future__ import annotations

from matplotlib.pyplot import show, subplots
from numpy import ones_like, triu
from seaborn import heatmap, set_palette

from ydata.report.style_guide import NOTEBOOK_FIG_SIZE, YDATA_COLORS, YDATA_HEATMAP_CMAP

set_palette(YDATA_COLORS)


def simulation_heatmaps(
    correlations: dict, title: str, suptitles: list, plt_show=False
):
    """Simulation report metrics correlation matrices."""
    assert len(correlations) == 4, "Please provide 4 correlation matrices."
    fig, ax = subplots(1, 4, figsize=NOTEBOOK_FIG_SIZE)
    for i, items in enumerate(correlations.items()):
        _, matrix = items
        _ = triu(ones_like(matrix, dtype=bool))
        heatmap(
            data=matrix,
            annot=True,
            fmt=".2f",
            cmap=YDATA_HEATMAP_CMAP,
            vmax=0.3,
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.5},
            ax=ax.flat[i],
        )
        ax.flat[i].set_title(suptitles[i], {"size": 10})
    fig.suptitle(title, fontsize=15, weight="bold")

    if plt_show:
        show()
    return fig.axes[0]
