from ydata.profiling.model import BaseDescription
from ydata_profiling.config import Settings as Settings
from ydata_profiling.report.presentation.core.renderable import Renderable as Renderable
from ydata_profiling.report.presentation.core.root import Root

def get_report_structure(config: Settings, summary: BaseDescription) -> Root:
    """Generate a HTML report from summary statistics and a given sample.

    Args:
      config: report Settings object
      summary: Statistics to use for the overview, variables, correlations and missing values.

    Returns:
      The profile report in HTML format
    """
def render_variables_section(config: Settings, dataframe_summary: BaseDescription) -> list:
    """Render the HTML for each of the variables in the DataFrame.

    Args:
        config: report Settings object
        dataframe_summary: The statistics for each variable.

    Returns:
        The rendered HTML, where each row represents a variable.
    """
