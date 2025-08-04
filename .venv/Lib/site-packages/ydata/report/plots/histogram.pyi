from pandas import DataFrame

def plot_hist(real_ts: DataFrame, synth_ts: DataFrame, title: str = '', x_label: str = '', y_label: str = '', save_path: str = None, plt_show: bool = False, variable: str = '0'):
    """Given two Pandas dataframes with real data and synthetic samples both
    histograms are plotted."""
def plot_period_hist(real_ts: list, synth_ts: list, periods: list, variable: str, title: str = '', xlab: str = '', save_path: str = None, plt_show: bool = False):
    """Given Pandas dataframes with real data and synthetic samples both
    histograms are plotted."""
