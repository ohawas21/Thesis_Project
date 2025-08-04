from _typeshed import Incomplete
from ydata_profiling.config import Settings as Settings
from ydata_profiling.model.typeset import ProfilingTypeSet as BaseTypeSet

class ProfilingTypeSet(BaseTypeSet):
    config: Incomplete
    type_schema: Incomplete
    def __init__(self, config: Settings, type_schema: dict = None) -> None: ...
