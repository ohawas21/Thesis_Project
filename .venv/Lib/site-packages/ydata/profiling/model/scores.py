"""
    File that computes the overall data quality scores for ydata-profiling
"""
from enum import Enum

from ydata_profiling.config import Settings
from ydata_profiling.model.alerts import skewness_alert

class ScoreType(Enum):
    completeness = "COMPLETENESS"
    accuracy = "ACCURACY"
    consistency = "CONSISTENCY"
    validity = "VALIDITY"
    duplication = "DUPLICATION"
    uniqueness = "UNIQUENESS"

def completeness_score(table_stats: dict,
                       series_description: dict,
                       variables: dict) -> float:
    """
    Args:
        table_stats: A dict with the table overall statistics
        series_description: A dict with the statistics computed per column
        variables: a dict with the dataset typeset

    Returns: A score between 0 and 1 for the completeness of the dataset.
    """
    numerical_cols=[col for col, type in variables.items() if type=='Numeric']

    p_missing_cels = 1-table_stats['p_cells_missing']

    if len(numerical_cols)>0:
        p_zeros = (sum([(1-series_description[col]['p_zeros']) for col in numerical_cols]))/len(numerical_cols)
        p_inf = (sum([(1-series_description[col]['p_infinite']) for col in numerical_cols]))/len(numerical_cols)

        return 0.8 * p_missing_cels + 0.05 * p_zeros + 0.15 * p_inf
    return p_missing_cels

def consistency_score(config: Settings,
                      series_description: dict,
                      variables: dict) -> float:

    categorical_cols = [col for col, type in variables.items() if type=='Categorical']
    numerical_cols = [col for col, type in variables.items() if type=='Numeric']
    date_cols = [col for col, type in variables.items() if type=='DateTime']
    boolean_cols = [col for col, type in variables.items() if type=='Boolean']

    score=0.0
    n_metrics=0

    if len(numerical_cols)>0:
        skewness_threshold = config.vars.num.skewness_threshold
        p_skewness = sum([skewness_alert(series_description[col]['skewness'], threshold=skewness_threshold)
                          for col in numerical_cols])/len(numerical_cols)
        score+=(1-p_skewness)
        n_metrics+=1

    if len(categorical_cols)>0:
        categorical_score=0.0
        imbalance_threshold = config.vars.cat.imbalance_threshold

        if config.vars.cat.dirty_categories:
            p_fuzzy_vals = sum([(1-series_description[col]['p_fuzzy_vals']) for col in categorical_cols])/len(categorical_cols)
            categorical_score+=p_fuzzy_vals

        p_imbalanced = sum([series_description[col]['imbalance']>imbalance_threshold for col in categorical_cols])/len(categorical_cols)
        categorical_score+=(1-p_imbalanced)

        score+=(categorical_score/2)
        n_metrics+=1

    if len(date_cols)>0:
        p_invalid_dates = sum([(1-series_description[col]['p_invalid_dates']) for col in date_cols])/len(date_cols)
        score+=p_invalid_dates
        n_metrics += 1


    if len(boolean_cols) > 0:
        imbalance_threshold = config.vars.bool.imbalance_threshold
        p_imbalanced = sum(
            [series_description[col]['imbalance'] > imbalance_threshold for col in boolean_cols]) / len(
            boolean_cols)
        score+=(1 - p_imbalanced)
        n_metrics += 1

    return score/n_metrics if n_metrics>0 else None

def duplication_score(table_stats: dict) -> float:
    """
    What information do I need to have this computed? I can receive floats onluy
    Args:
        table_stats: A dict with the table overall statistics
    Returns:
    """
    return 0.6 * (1-table_stats['p_duplicates']) + 0.4 * (1-table_stats['p_near_dups'])

def uniqueness_score(stats: dict) -> float:
    return

def describe_scores(config: Settings,
                    table_stats: dict,
                    variables: dict,
                    series_description: dict) -> dict:
    """
    Calculates different scores depending on the calculated profiling metrics and statistics
    Args:
        table_stats: A dictionary containing table specific statistics
        series_description: A dictionary with the statistics per dataset variable
        correlations: A dictionary with the selected correlation matrices # check whether this can be optional

    Returns: A dictionary with all the scores computed.
    """

    p_completeness = completeness_score(table_stats=table_stats,
                                        series_description=series_description,
                                        variables=variables)

    if config.duplicates.head > 0:
        p_duplicates = duplication_score(table_stats=table_stats)
    else:
        p_duplicates = None

    p_consistency = consistency_score(config=config,
                                      series_description=series_description,
                                      variables=variables)

    metrics = [p_duplicates, p_consistency, p_completeness]
    metrics = [metric for metric in metrics if metric is not None]
    total_metrics, n_metrics = sum(metrics), len(metrics)

    return {
            'overall_score': total_metrics / n_metrics,
            'completeness': p_completeness,
            'consistency': p_consistency,
            'uniqueness': p_duplicates}
