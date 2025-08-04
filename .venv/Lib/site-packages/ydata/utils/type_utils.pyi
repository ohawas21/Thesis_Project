from decimal import Context

def float_to_str(f: float, ctx: Context) -> str:
    """Convert a float to a decimal string representation.

    Small float will be usually represented by Python using scientific
    notation. However, we might need a decimal representation in some
    cases.
    """
