"""YData Profiling module."""
import os

os.environ['YDATA_SUPPRESS_BANNER']='True'

from ydata.profiling.profile_report import ProfileReport

__all__ = ["ProfileReport"]
