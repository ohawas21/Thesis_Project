from matplotlib.pyplot import savefig, show, subplots, suptitle, tight_layout, xlabel
from pandas import DataFrame, Series
from scipy.stats import kurtosis, skew

from ydata.report.style_guide import NOTEBOOK_FIG_SIZE


def plot_hist(
    real_ts: DataFrame,
    synth_ts: DataFrame,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    save_path: str = None,
    plt_show: bool = False,
    variable: str = "0",
):
    """Given two Pandas dataframes with real data and synthetic samples both
    histograms are plotted."""

    assert isinstance(real_ts, DataFrame) or isinstance(
        real_ts, Series
    ), "Real data must be a pandas DataFrame or Series."
    assert isinstance(synth_ts, DataFrame) or isinstance(
        synth_ts, Series
    ), "Synth data must be a pandas DataFrame or Series."

    fig, axs = subplots(1, 1, sharex=True, figsize=NOTEBOOK_FIG_SIZE)

    axs.hist(
        synth_ts,
        label="Synth",
        bins=50,
        alpha=0.2,
        density=True,
        histtype="barstacked",
        color="red",
    )
    synth_ts.plot.kde(ax=axs, legend=False, label="_nolegend_", color="red")

    axs.hist(
        real_ts,
        label="Real",
        bins=50,
        alpha=0.2,
        density=True,
        histtype="barstacked",
        color="grey",
    )
    real_ts.plot.kde(ax=axs, legend=False, label="_nolegend_", color="grey")

    axs.spines["top"].set_visible(False)
    axs.spines["right"].set_visible(False)
    axs.spines["bottom"].set_visible(False)
    axs.spines["left"].set_visible(False)
    axs.legend(loc="upper right")

    textstr = "\n".join(
        (
            r"%s" % ("- Real",),
            r"$skewness=%.2f$" % (skew(real_ts),),
            r"$kurtosis=%.2f$" % (kurtosis(real_ts),),
        )
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    axs.text(
        0.025,
        0.95,
        textstr,
        fontsize=14,
        verticalalignment="top",
        bbox=props,
        transform=axs.transAxes,
    )
    textstr = "\n".join(
        (
            r"%s" % ("- Synthetic",),
            r"$skewness=%.2f$" % (skew(synth_ts),),
            r"$kurtosis=%.2f$" % (kurtosis(synth_ts),),
        )
    )
    axs.text(
        0.025,
        0.8,
        textstr,
        fontsize=14,
        verticalalignment="top",
        bbox=props,
        transform=axs.transAxes,
    )
    axs.set_xlabel(x_label, fontsize=12)
    axs.set_ylabel(y_label, fontsize=12)
    suptitle(title, fontsize=16)
    tight_layout()
    if save_path is not None:
        fig_name = title.replace(" ", "_") + ".png"
        savefig("/".join((save_path, fig_name)))
    if plt_show is True:
        show()
    return axs


def plot_period_hist(
    real_ts: list,
    synth_ts: list,
    periods: list,
    variable: str,
    title: str = "",
    xlab: str = "",
    save_path: str = None,
    plt_show: bool = False,
):
    """Given Pandas dataframes with real data and synthetic samples both
    histograms are plotted."""

    assert (
        isinstance(synth_ts, list) and len(synth_ts) > 0
    ), "The input data must be a valid list of synth samples."
    assert isinstance(
        periods, list
    ), "The periods of each sample must be a list of periods."
    assert len(synth_ts) == len(
        periods
    ), "The length and order of the synth samples list must match the periods list."
    assert len(synth_ts) == len(
        real_ts
    ), "The length of synth samples list must match the real samples list."

    n_plots = len(synth_ts)
    if n_plots > 1:
        if n_plots % 2 == 0:
            n_cols = 2
        else:
            n_cols = 3
        n_rows = n_plots // n_cols
    else:
        n_rows = 1
        n_cols = 1

    fig, axs = subplots(n_rows, n_cols, figsize=NOTEBOOK_FIG_SIZE)
    p = 0
    for k in range(n_rows):
        for j in range(n_cols):
            axs[k][j].hist(
                synth_ts[p][variable],
                label="Synth",
                bins=50,
                density=True,
                histtype="barstacked",
                alpha=0.2,
                color="red",
            )
            synth_ts[p][variable].plot.kde(
                ax=axs[k][j], legend=False, label="_nolegend_", color="red"
            )
            axs[k][j].hist(
                real_ts[p][variable],
                label="Real",
                bins=50,
                density=True,
                histtype="barstacked",
                alpha=0.2,
                color="grey",
            )
            real_ts[p][variable].plot.kde(
                ax=axs[k][j], legend=False, label="_nolegend_", color="black"
            )
            axs[k][j].legend(loc="upper right")

            textstr = "\n".join(
                (
                    r"%s" % ("- Real",),
                    r"$skewness=%.2f$" % (skew(real_ts[p][variable]),),
                    r"$kurtosis=%.2f$" % (kurtosis(real_ts[p][variable]),),
                )
            )
            props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
            axs[k][j].text(
                0.025,
                0.95,
                textstr,
                fontsize=14,
                verticalalignment="top",
                bbox=props,
                transform=axs[k][j].transAxes,
            )
            textstr = "\n".join(
                (
                    r"%s" % ("- Synthetic",),
                    r"$skewness=%.2f$" % (skew(synth_ts[p][variable]),),
                    r"$kurtosis=%.2f$" % (kurtosis(synth_ts[p][variable]),),
                )
            )
            axs[k][j].text(
                0.025,
                0.65,
                textstr,
                fontsize=14,
                verticalalignment="top",
                bbox=props,
                transform=axs[k][j].transAxes,
            )
            axs[k][j].title.set_text(periods[p])
            p += 1

    suptitle(title + f" of {variable}", fontsize=16)
    if xlab != "":
        xlabel(xlab, fontsize=14)

    if save_path is not None:
        fig_name = title.replace(" ", "_") + ".png"
        savefig("/".join((save_path, fig_name)))
    if plt_show is True:
        show()
    return axs
