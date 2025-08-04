from matplotlib.pyplot import savefig, show, subplots, suptitle
from pandas import DataFrame, Series
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from ydata.report.style_guide import NOTEBOOK_FIG_SIZE


def compare_acf(
    real_ts: DataFrame,
    synth_ts: DataFrame,
    n_lags: int = 24,
    title: str = "Autocorrelation Functions",
    variable: str = "0",
    save_path=None,
    plt_show: bool = False,
):
    """Given two pandas dataframes the autocorrelation function is plotted."""

    assert isinstance(real_ts, DataFrame) or isinstance(
        real_ts, Series
    ), "Real data must be a pandas DataFrame or Series."
    assert isinstance(synth_ts, DataFrame) or isinstance(
        synth_ts, Series
    ), "Synth data must be a pandas DataFrame or Series."

    fig, axs = subplots(1, 2, sharex=True, figsize=NOTEBOOK_FIG_SIZE)
    plot_acf(real_ts, ax=axs[0], lags=n_lags, title=f"Real - {variable}")
    plot_acf(synth_ts, ax=axs[1], lags=n_lags, title=f"Synthetic - {variable}")
    suptitle("Autocorrelation Function", fontsize=16)

    if save_path is not None:
        fig_name = title.replace(" ", "_") + ".png"
        savefig("/".join((save_path, fig_name)))
    if plt_show is True:
        show()
    return axs


def compare_pacf(
    real_ts: DataFrame,
    synth_ts: DataFrame,
    n_lags: int = 24,
    title: str = "Partial Autocorrelation Functions",
    variable: str = "0",
    save_path=None,
    plt_show: bool = False,
):
    """Given two pandas dataframes the partial autocorrelation function is
    plotted."""

    assert isinstance(real_ts, DataFrame) or isinstance(
        real_ts, Series
    ), "Real data must be a pandas DataFrame or Series."
    assert isinstance(synth_ts, DataFrame) or isinstance(
        synth_ts, Series
    ), "Synth data must be a pandas DataFrame or Series."

    fig, axs = subplots(1, 2, sharex=True, figsize=NOTEBOOK_FIG_SIZE)
    plot_pacf(real_ts, ax=axs[0], lags=n_lags, title=f"Real - {variable}")
    plot_pacf(synth_ts, ax=axs[1], lags=n_lags,
              title=f"Synthetic - {variable}")
    suptitle("Partial Autocorrelation Function", fontsize=16)

    if save_path is not None:
        fig_name = title.replace(" ", "_") + ".png"
        savefig("/".join((save_path, fig_name)))
    if plt_show is True:
        show()
    return axs
