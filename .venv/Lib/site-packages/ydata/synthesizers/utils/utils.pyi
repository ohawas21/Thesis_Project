def check_special_characters(val_list): ...
def check_for_letters(val_list): ...
def to_numpy(x):
    """Casts torch.Tensor to a numpy ndarray.

    The function detaches the tensor from its gradients, then puts it
    onto the cpu and at last casts it to numpy.
    """
