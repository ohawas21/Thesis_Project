from ydata.__models._cartmodel.maps import Smoothing

def visitor_seq_to_predictor_mat(visit_sequence: list[str]):
    """Generate a predictor matrix based on a visitor sequence.

    Args:
        visit_sequence (List[str]): visitor sequence used to construct the predictor matrix

    Returns:
        pd.DataFrame: Predictor matrix
    """
def intialize_smoothing_strategy(smoothing_method: Smoothing, columns: list[str]):
    """Define the smoothing strategy per column depending on a smoothing
    method.

    Args:
        smoothing_method (Smoothing): Smoothing strategy
        columns (List[str]): List of columns on which Smoothing needs to be applied

    Returns:
        Dict[str, bool]: Dictionary indicating if a column needs to be smoothed or not
    """
def validate_datatypes(enabled_types, columns): ...
