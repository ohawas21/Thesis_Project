from copy import deepcopy
from dataclasses import dataclass, fields

from ydata.core.enum import StringEnum


class ComputeMode(StringEnum):
    NOW = "now"
    LATER = "later"
    DEFERRED = "deferred"
    AUTO = "auto"


def init_compute_mode(mode: ComputeMode | bool | str | None) -> ComputeMode:
    if isinstance(mode, ComputeMode):
        return mode
    elif mode is None:
        return ComputeMode.AUTO
    elif isinstance(mode, bool):
        return ComputeMode.NOW if mode else ComputeMode.LATER
    else:
        return ComputeMode(mode)


@dataclass
class ComputeConfig:
    characteristics: ComputeMode = ComputeMode.AUTO
    correlation: ComputeMode = ComputeMode.AUTO
    interaction: ComputeMode = ComputeMode.AUTO

    pairwise_metrics: bool = True
    infer_characteristics: bool = False

    def __post_init__(self):
        for f in fields(self):
            if f.type == ComputeMode:
                mode = getattr(self, f.name)
                setattr(self, f.name, init_compute_mode(mode))

    def resolve_auto(self, inplace: bool = False) -> "ComputeConfig":
        c = self if inplace else deepcopy(self)
        if c.characteristics == ComputeMode.AUTO:
            c.characteristics = ComputeMode.NOW if c.infer_characteristics else ComputeMode.LATER
        if c.correlation == ComputeMode.AUTO:
            c.correlation = ComputeMode.NOW if c.pairwise_metrics else ComputeMode.LATER
        if c.interaction == ComputeMode.AUTO:
            c.interaction = ComputeMode.LATER
        return c
