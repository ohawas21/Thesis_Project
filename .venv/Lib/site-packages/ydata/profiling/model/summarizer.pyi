from dataclasses import asdict as asdict
from visions import VisionsTypeset as VisionsTypeset
from ydata_profiling.config import Settings as Settings
from ydata_profiling.model import BaseDescription as BaseDescription
from ydata_profiling.model.summarizer import ProfilingSummarizer

class YDataProfilingSummarizer(ProfilingSummarizer):
    """The default YData Profiling summarizer"""
