"""Some utils functions."""


def check_special_characters(val_list):
    import re

    regex = re.compile(r"[@_!#$%^&*()<>?/\|}{~:]")
    result = False
    for val in val_list:
        if regex.search(str(val[0])) is not None:
            result = True
            break
    return result


def check_for_letters(val_list):
    import re

    regex = re.compile(r"(?i)[a-z]")
    result = False
    for val in val_list:
        if regex.search(str(val[0])) is not None:
            result = True
            break
    return result


def to_numpy(x):
    """Casts torch.Tensor to a numpy ndarray.

    The function detaches the tensor from its gradients, then puts it
    onto the cpu and at last casts it to numpy.
    """
    return x.detach().cpu().numpy()
