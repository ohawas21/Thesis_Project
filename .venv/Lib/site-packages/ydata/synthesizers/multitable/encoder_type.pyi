from ydata.core.enum import StringEnum

class EncoderType(StringEnum):
    """Encoder type used for joint table parents."""
    AFFINITY_PROPAGATION: str
    BIRCH: str
    MEAN_SHIFT: str
