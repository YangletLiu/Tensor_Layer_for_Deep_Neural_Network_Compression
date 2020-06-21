import tensorly as tl
import torch
import numpy as np


def tensor_ring(input_tensor, rank):
    '''Tensor ring (TR) decomposition via recursive SVD

        Decomposes input_tensor into a sequence of order-3 tensors (factors),
        with the input rank of the first factor equal to the output rank of the 
        last factor. This code is modified from MPS decomposition in tensorly 
        lib's src code

    Parameters
    ----------
    input_tensor : tensorly.tensor
    rank : {int, int list}
            maximum allowable rank of the factors
            if int, then this is the same for all the factors
            if int list, then rank[k] is the rank of the kth factor

    Returns
    -------
    factors : Tensor ring factors
              order-3 tensors of the tensor ring decomposition
    '''

    # Check user input for errors
    tensor_size = input_tensor.shape
    n_dim = len(tensor_size)

    if isinstance(rank, int):
        rank = [rank] * n_dim
    elif n_dim != len(rank):
        message = 'Provided incorrect number of ranks. '
        raise(ValueError(message))
    rank = list(rank)

    # Initialization
    unfolding = tl.unfold(input_tensor, 0)
    factors = [None] * n_dim
    U, S, V = tl.partial_svd(unfolding, rank[0])
    r0 = int(np.sqrt(rank[0]))
    while rank[0] % r0:
        r0 -= 1;
    T0 = tl.reshape(U, (tensor_size[0], r0, rank[0] // r0))
    factors[0] = torch.transpose(torch.tensor(T0), 0, 1)
    unfolding = tl.reshape(S, (-1, 1)) * V
    rank[1] = rank[0] // r0
    rank.append(r0)
    
    # Getting the MPS factors up to n_dim
    for k in range(1, n_dim):

        # Reshape the unfolding matrix of the remaining factors
        n_row = int(rank[k]*tensor_size[k])
        unfolding = tl.reshape(unfolding, (n_row, -1))

        # SVD of unfolding matrix
        (n_row, n_column) = unfolding.shape
        rank[k+1] = min(n_row, n_column, rank[k+1])
        U, S, V = tl.partial_svd(unfolding, rank[k+1])

        # Get kth MPS factor
        factors[k] = tl.reshape(U, (rank[k], tensor_size[k], rank[k+1]))

        # Get new unfolding matrix for the remaining factors
        unfolding = tl.reshape(S, (-1, 1)) * V
    
    return factors