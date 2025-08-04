def compute_pairwise_distances(x, y):
    """Computes the squared pairwise Euclidean distances between x and y.

    Args:
      x: a tensor of shape [num_x_samples, num_features]
      y: a tensor of shape [num_y_samples, num_features]
    Returns:
      a distance matrix of dimensions [num_x_samples, num_y_samples].
    Raises:
      ValueError: if the inputs do no matched the specified dimensions.
    """
def gaussian_kernel_matrix(x, y, sigmas):
    """Computes a Guassian Radial Basis Kernel between the samples of x and y.

    Create a sum of multiple gaussian kernels each having a width sigma_i.
     Args:
       x: a tensor of shape [num_samples, num_features]
       y: a tensor of shape [num_samples, num_features]
       sigmas: a tensor of floats which denote the widths of each of the
         gaussians in the kernel.
     Returns:
       A tensor of shape [num_samples{x}, num_samples{y}] with the RBF kernel.
    """
def maximum_mean_discrepancy(x, y, kernel):
    """Computes the Maximum Mean Discrepancy (MMD) of two samples: x and y.
    Maximum Mean Discrepancy (MMD) is a distance-measure between the samples of
    the distributions of x and y. Here we use the kernel two sample estimate
    using the empirical mean of the two distributions.

    MMD^2(P, Q) = || \\E{\\phi(x)} - \\E{\\phi(y)} ||^2
                = \\E{ K(x, x) } + \\E{ K(y, y) } - 2 \\E{ K(x, y) },
    where K = <\\phi(x), \\phi(y)>,
      is the desired kernel function, in this case a radial basis kernel.
    Args:
        x: a tensor of shape [num_samples, num_features]
        y: a tensor of shape [num_samples, num_features]
        kernel: a function which computes the kernel in MMD. Defaults to the
                GaussianKernelMatrix.
    Returns:
        a scalar denoting the squared maximum mean discrepancy loss.
    """
def mmd_loss(source_samples, target_samples, weight: float = 1.0):
    """Adds a similarity loss term, the MMD between two representations.

    This Maximum Mean Discrepancy (MMD) loss is calculated with a number of
    different Gaussian kernels.
    Args:
      source_samples: a tensor of shape [num_samples, num_features].
      target_samples: a tensor of shape [num_samples, num_features].
      weight: the weight of the MMD loss.
      scope: optional name scope for summary tags.
    Returns:
      a scalar tensor representing the MMD loss value.
    """
