from ydata.core.enum import StringEnum


class EncoderType(StringEnum):
    """Encoder type used for joint table parents."""
    AFFINITY_PROPAGATION = "AFFINITY_PROPAGATION"
    BIRCH = "BIRCH"
    MEAN_SHIFT = "MEAN_SHIFT"

    def __str__(self):
        if self in set(EncoderType):
            return " ".join(self.value.split("_")).title()
        return "N/D"
