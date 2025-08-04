import pandas as pd
from typing import Any, Sequence
from ydata_profiling.config import Settings as Settings

def normalize_text(text):
    """Improved normalization function for both text and numeric data."""
def get_potential_duplicates(config: Settings, df: pd.DataFrame, supported_columns: Sequence) -> pd.DataFrame | None:
    """
    Detects near duplicate rows using fuzzy matching and similarity scores
    Args:
        df: The original dataframe to be searched/validated

    Returns: a pandas Dataframe with the near duplicate records and respective percentages
    """
def get_duplicates(config: Settings, df: pd.DataFrame, len_df: int) -> tuple[dict[str, Any], pd.DataFrame | None]: ...
def get_near_duplicates(config: Settings, df: pd.DataFrame, len_df: int) -> tuple[dict[str, Any], pd.DataFrame | None]: ...
