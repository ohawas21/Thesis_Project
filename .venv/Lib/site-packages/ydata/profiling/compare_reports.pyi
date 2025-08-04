from asn1crypto.cms import V2Form as V2Form
from ydata.profiling.model import BaseDescription
from ydata.profiling.profile_report import ProfileReport
from ydata_profiling.config import Settings
from ydata_profiling.model.alerts import Alert as Alert

def validate_reports(reports: list[ProfileReport] | list[BaseDescription], configs: list[dict]) -> None:
    """Validate if the reports are comparable.

    Args:
        reports: two reports to compare
                 input may either be a ProfileReport, or the summary obtained from report.get_description()
    """
def compare(reports: list[ProfileReport] | list[BaseDescription], config: Settings | None = None, compute: bool = False) -> ProfileReport:
    """
    Compare Profile reports

    Args:
        reports: two reports to compare
                 input may either be a ProfileReport, or the summary obtained from report.get_description()
        config: the settings object for the merged ProfileReport
        compute: recompute the profile report using config or the left report config
                 recommended in cases where the reports were created using different settings

    """
