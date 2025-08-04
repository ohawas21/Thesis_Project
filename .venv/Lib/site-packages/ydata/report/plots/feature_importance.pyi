def visualize_features(best_features_dataset) -> None:
    """Plot a visualization of the top real vs synthetic features.

    Usage:
    >>> from ydata.report.metrics._metrics import feature_importance
    >>> best_features = feature_importance(df_real, df_synth, target='class')
    >>> visualize_features(best_features)
    """
