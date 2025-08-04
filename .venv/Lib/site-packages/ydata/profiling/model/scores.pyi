from enum import Enum
from ydata_profiling.config import Settings as Settings

class ScoreType(Enum):
    completeness = 'COMPLETENESS'
    accuracy = 'ACCURACY'
    consistency = 'CONSISTENCY'
    validity = 'VALIDITY'
    duplication = 'DUPLICATION'
    uniqueness = 'UNIQUENESS'

def completeness_score(table_stats: dict, series_description: dict, variables: dict) -> float:
    """
    Args:
        table_stats: A dict with the table overall statistics
        series_description: A dict with the statistics computed per column
        variables: a dict with the dataset typeset

    Returns: A score between 0 and 1 for the completeness of the dataset.
    """
def consistency_score(config: Settings, series_description: dict, variables: dict) -> float: ...
def duplication_score(table_stats: dict) -> float:
    """
    What information do I need to have this computed? I can receive floats onluy
    Args:
        table_stats: A dict with the table overall statistics
    Returns:
    """
def uniqueness_score(stats: dict) -> float: ...
def describe_scores(config: Settings, table_stats: dict, variables: dict, series_description: dict) -> dict:
    """
    Calculates different scores depending on the calculated profiling metrics and statistics
    Args:
        table_stats: A dictionary containing table specific statistics
        series_description: A dictionary with the statistics per dataset variable
        correlations: A dictionary with the selected correlation matrices # check whether this can be optional

    Returns: A dictionary with all the scores computed.
    """
