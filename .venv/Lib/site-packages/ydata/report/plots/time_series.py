from matplotlib.dates import DateFormatter
from matplotlib.pyplot import savefig, set_loglevel, show, subplots, suptitle
from matplotlib.ticker import IndexLocator
from pandas import DataFrame, DatetimeIndex, Series

from ydata.report.style_guide import NOTEBOOK_FIG_SIZE, YDATA_CMAP_COLORS

set_loglevel("WARNING")


def plot_timeseries(
    real_ts: DataFrame,
    synth_ts: DataFrame,
    variable: str = "0",
    title: str = "Real and Synthetic data.",
    save_path=None,
    plt_show: bool = False,
):
    """Lineplot of real data vs synthetic data."""

    assert isinstance(real_ts, DataFrame) or isinstance(
        real_ts, Series
    ), "Real data must be a pandas DataFrame or Series."
    assert isinstance(synth_ts, DataFrame) or isinstance(
        synth_ts, Series
    ), "Synth data must be a pandas DataFrame or Series."

    fig, axs = subplots(1, 1, sharex=True, figsize=NOTEBOOK_FIG_SIZE)
    a = axs
    a.plot(real_ts, label="Real")
    a.plot(synth_ts)
    a.legend(loc="upper right")
    a.set_title(f"Feature: {variable}", fontsize=14)
    suptitle("Real and Synth Time series", fontsize=16)
    if save_path is not None:
        fig_name = title.replace(" ", "_") + ".png"
        savefig("/".join((save_path, fig_name)))
    if plt_show is True:
        show()
    return a


def plot_samples(
    real_ts: DataFrame,
    synth_ts: list,
    variable: str = "0",
    title: str = "Synthetic Samples",
    save_path=None,
    x_axis=None,
    x_label: str = "",
    y_label: str = "",
    plt_show: bool = False,
):
    """Plot all samples for each sequence of real data."""

    assert isinstance(real_ts, DataFrame) or isinstance(
        real_ts, Series
    ), "Shape of input samples must be 2d (ts_dim, n_samples)"
    assert isinstance(
        synth_ts, list
    ), "Synth data should be a list of univariate DataFrames."

    # Customize x_axis
    if x_axis is not None:
        assert (
            len(x_axis) == synth_ts[0].shape[0]
        ), "The provided x_axis length must match the synthetic samples."
    else:
        x_axis = synth_ts[0].index

    fig, ax = subplots(1, 1, sharex=True, figsize=NOTEBOOK_FIG_SIZE)

    if isinstance(x_axis, DatetimeIndex):
        locator = IndexLocator(base=1, offset=0)
        datefmt = DateFormatter("%Y-%m-%d")
        ax.xaxis.set_major_formatter(datefmt)
    else:
        locator = IndexLocator(base=1, offset=0)

    ax.set_prop_cycle(color=YDATA_CMAP_COLORS)
    real_ts.index = x_axis
    ax.plot(real_ts, label="Real", linewidth=4)
    n_samples = len(synth_ts)
    for s in range(n_samples):
        synth_ts[s].index = x_axis
        if s == 0:
            ax.plot(synth_ts[s], label="Synth", linestyle="dashed")
        else:
            ax.plot(synth_ts[s], linestyle="dashed")
        ax.legend(loc="upper right")
        # Only show every n values in x-axis
        ax.xaxis.set_major_locator(locator)

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    suptitle(title, fontsize=16)
    # If rotation is needed
    # fig.autofmt_xdate()

    if save_path is not None:
        fig_name = title.replace(" ", "_") + ".png"
        savefig("/".join((save_path, fig_name)))
    if plt_show is True:
        show()
    return ax
