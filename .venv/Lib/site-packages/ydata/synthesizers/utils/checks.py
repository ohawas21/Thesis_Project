from __future__ import absolute_import, division, print_function

from numpy import ndarray


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

    def decorated(self, data, *args, **kwargs):
        if not (
            isinstance(data, ndarray) and len(
                data.shape) == 2 and data.shape[1] == 1
        ):
            raise ValueError(
                "The argument `data` must be a numpy.ndarray with shape (n, 1)."
            )

        return function(self, data, *args, **kwargs)

    decorated.__doc__ = function.__doc__
    return decorated
