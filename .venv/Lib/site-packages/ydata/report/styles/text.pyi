from _typeshed import Incomplete

class TextFormatter:
    @staticmethod
    def int_to_string(n: int):
        """Returns formatted thousands.

        >>> SyntheticDataProfile._int_to_string(1_200)
        '1K'
        >>> SyntheticDataProfile._int_to_string(700)
        '700'
        >>> SyntheticDataProfile._int_to_string(10_700)
        '11K'
        """
    @staticmethod
    def float_to_pct(n: float, decimals: int = 1):
        """Returns formatted percentages.

        >>> SyntheticDataProfile._float_to_pct(0.2)
        '20.0%'
        >>> SyntheticDataProfile._float_to_pct(25)
        '2500.0%'
        >>> SyntheticDataProfile._float_to_pct(-0.02)
        '-2.0%'
        """
    @staticmethod
    def format_float(n: float, decimals: Incomplete | None = None):
        """Returns formatted floats.

        >>> SyntheticDataProfile._format_float(2.14)
        '2.1'
        >>> SyntheticDataProfile._format_float(438.83)
        '439'
        >>> SyntheticDataProfile._format_float(-0.02)
        '-0.02'
        """
