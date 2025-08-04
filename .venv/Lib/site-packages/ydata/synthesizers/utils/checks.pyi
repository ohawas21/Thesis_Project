def check_inputs(function):
    """Validate inputs for functions whose first argument is a numpy.ndarray
    with shape (n,1).

    Args:
        function(callable): Method to validate.
    Returns:
        callable: Will check the inputs before calling :attr:`function`.
    Raises:
        ValueError: If first argument is not a valid :class:`numpy.array` of shape (n, 1).
    """
