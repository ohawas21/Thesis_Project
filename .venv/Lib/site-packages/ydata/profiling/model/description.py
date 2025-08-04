from dataclasses import dataclass
from typing import Any

from ydata_profiling.model.description import BaseDescription as Base


@dataclass
class BaseDescription(Base):
    """Description of DataFrame.

    Attributes:
        analysis (BaseAnalysis): Base info about report. Title, start time and end time of description generating.
        table (Any): DataFrame statistic. Base information about DataFrame.
        variables (Dict[str, Any]): Description of variables (columns) of DataFrame. Key is column name, value is description dictionary.
        scatter (Any): Pairwise scatter for all variables. Plot interactions between variables.
        correlations (Dict[str, Any]): Prepare correlation matrix for DataFrame
        missing (Dict[str, Any]): Describe missing values.
        alerts (Any): Take alerts from all modules (variables, scatter, correlations), and group them.
        package (Dict[str, Any]): Contains version of pandas profiling and config.
        sample (Any): Sample of data.
        duplicates (Any): Description of duplicates.
        near_duplicates (Any): Description of near duplicate rows in the dataset
        outliers (Any): Description of data outliers.
    """
    near_duplicates: Any = None
    outliers: Any = None
    scores: Any = None
