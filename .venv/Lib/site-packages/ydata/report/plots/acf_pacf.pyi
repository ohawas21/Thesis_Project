from _typeshed import Incomplete
from pandas import DataFrame

def compare_acf(real_ts: DataFrame, synth_ts: DataFrame, n_lags: int = 24, title: str = 'Autocorrelation Functions', variable: str = '0', save_path: Incomplete | None = None, plt_show: bool = False):
    """Given two pandas dataframes the autocorrelation function is plotted."""
def compare_pacf(real_ts: DataFrame, synth_ts: DataFrame, n_lags: int = 24, title: str = 'Partial Autocorrelation Functions', variable: str = '0', save_path: Incomplete | None = None, plt_show: bool = False):
    """Given two pandas dataframes the partial autocorrelation function is
    plotted."""
