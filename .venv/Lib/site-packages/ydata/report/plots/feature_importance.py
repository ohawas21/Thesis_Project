"""Methods to display synthetic data quality insights."""
import matplotlib.pyplot as plt
import seaborn as sns


def visualize_features(best_features_dataset):
    """Plot a visualization of the top real vs synthetic features.

    Usage:
    >>> from ydata.report.metrics._metrics import feature_importance
    >>> best_features = feature_importance(df_real, df_synth, target='class')
    >>> visualize_features(best_features)
    """
    fig, ax = plt.subplots()  # noqa: F841
    ax = sns.histplot(  # noqa: F841
        best_features_dataset,
        x="columns",
        hue="type",
        weights="score",
        multiple="dodge",
        shrink=0.8,
    )
    plt.xticks(rotation=30)
    plt.ylabel("Score")
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], [
               "0%", "20%", "40%", "60%", "80%", "100%"])
