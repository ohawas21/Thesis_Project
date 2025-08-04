from _typeshed import Incomplete
from pandas import DataFrame

def plot_timeseries(real_ts: DataFrame, synth_ts: DataFrame, variable: str = '0', title: str = 'Real and Synthetic data.', save_path: Incomplete | None = None, plt_show: bool = False):
    """Lineplot of real data vs synthetic data."""
def plot_samples(real_ts: DataFrame, synth_ts: list, variable: str = '0', title: str = 'Synthetic Samples', save_path: Incomplete | None = None, x_axis: Incomplete | None = None, x_label: str = '', y_label: str = '', plt_show: bool = False):
    """Plot all samples for each sequence of real data."""
