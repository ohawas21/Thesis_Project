import numpy as np


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
        if n >= 1_000:
            return str(round(n / 1000)) + "K"
        else:
            return str(n)

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
        return f"{n*100:.{decimals}f}%"

    @staticmethod
    def format_float(n: float, decimals=None):
        """Returns formatted floats.

        >>> SyntheticDataProfile._format_float(2.14)
        '2.1'
        >>> SyntheticDataProfile._format_float(438.83)
        '439'
        >>> SyntheticDataProfile._format_float(-0.02)
        '-0.02'
        """
        if np.isnan(n):
            return "N/A"
        if abs(n) > 1e9:
            n_, unit = n / 1e9, "B"
        elif abs(n) > 1e6:
            n_, unit = n / 1e6, "M"
        elif abs(n) > 1e3:
            n_, unit = n / 1e3, "K"
        else:
            n_, unit = n, ""
        if decimals is None:
            decimals = 0 if abs(n) > 10 else 1 if abs(n) > 1 else 2
        return f"{n_:.{decimals}f}{unit}"
