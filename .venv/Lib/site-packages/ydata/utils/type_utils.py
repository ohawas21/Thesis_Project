from decimal import Context


def float_to_str(f: float, ctx: Context) -> str:
    """Convert a float to a decimal string representation.

    Small float will be usually represented by Python using scientific
    notation. However, we might need a decimal representation in some
    cases.
    """
    r = format(Context().create_decimal(repr(f)), 'f')
    if r.replace('.', '', 1).isdigit():  # Ensure that the float is not NaN, Infinity or other
        return r
    else:
        return '0.0'
