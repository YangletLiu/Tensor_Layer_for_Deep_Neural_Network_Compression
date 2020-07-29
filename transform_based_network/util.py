import sys
sys.path.append('../')
from common import *
from transform_based_network import *

import torch
import torch_dct as dct
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import StepLR


def bcirc(A):
    l, m, n = A.shape
    bcirc_A = []
    for i in range(l):
        bcirc_A.append(torch.roll(A, shifts=i, dims=0))
    return torch.cat(bcirc_A, dim=2).reshape(l*m, l*n)

def hankel(A):
    l, m, n = A.shape
    circ = torch.zeros(2 * l + 1, m, n)
    circ[l, ...] = torch.zeros(m, n)
    for i in range(l):
        k = circ.shape[0] - i - 1
        circ[i, ...] = A[i, ...]
        circ[k, ...] = A[i, ...]
    hankel_A = []
    for i in range(1, l + 1):
        hankel_A.append(circ[i : i + l, ...])
    return torch.cat(hankel_A, dim=2).reshape(l*m, l*n)

def toeplitz(A):
    l, m, n = A.shape
    circ = torch.zeros(2 * l - 1, m, n)
    for i in range(l):
        circ[i + l - 1, ...] = A[i, ...]
        circ[l - i - 1, ...] = A[i, ...]
    toeplitz_A = []
    for i in range(l):
        toeplitz_A.append(circ[l - i - 1: 2 * l - i - 1, ...])
    return torch.cat(toeplitz_A, dim=2).reshape(l*m, l*n)

def tph(A):
    #device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return torch.Tensor(toeplitz(A) + hankel(A))#.to(device)

def shift(X):
    A = X.clone()
    for i in range(1, A.shape[0]):
        k = A.shape[0] - i - 1
        A[k, ...] -= A[k + 1, ...]
    return A

def t_product(A, B):
    assert(A.shape[0] == B.shape[0] and A.shape[2] == B.shape[1])
    prod = torch.mm(tph(shift(A)), bcirc(B)[:, 0:B.shape[2]])
    return prod.reshape(A.shape[0], A.shape[1], B.shape[2])

def t_product_v2(A, B):
    assert(A.shape[0] == B.shape[0] and A.shape[2] == B.shape[1])
    dct_C = torch.zeros(A.shape[0], A.shape[1], B.shape[2])
    dct_A = torch_apply(dct.dct, A)
    dct_B = torch_apply(dct.dct, B)
    for k in range(A.shape[0]):
        dct_C[k, ...] = torch.mm(dct_A[k, ...], dct_B[k, ...])
    return torch_apply(dct.idct, dct_C)

def t_product_fft(A, B):
    assert(A.shape[0] == B.shape[0] and A.shape[2] == B.shape[1])
    prod = torch.mm(bcirc(A), bcirc(B)[:, 0:B.shape[2]])
    return prod.reshape(A.shape[0], A.shape[1], B.shape[2])

def t_product_fft_v2(A, B):
    assert(A.shape[0] == B.shape[0] and A.shape[2] == B.shape[1])
    dct_C = np.zeros((A.shape[0], A.shape[1], B.shape[2]), dtype=complex)
    for k in range(A.shape[0]):
        dct_C[k, ...] = np.fft.fft(A, axis=0)[k, ...] @ np.fft.fft(B, axis=0)[k, ...]
    return np.real(np.fft.ifft(dct_C, axis=0))

def scalar_tubal_func(output_tensor):
    l, m, n = output_tensor.shape
    lateral_slices = [output_tensor[:, :, i].reshape(l, m, 1) for i in range(n)]
    h_slice = []
    for s in lateral_slices:
        h_slice.append(h_func_dct(s))
    pro_matrix = torch.stack(h_slice, dim=2)
    return pro_matrix.reshape(m, n)

def h_func_dct(lateral_slice):
    l, m, n = lateral_slice.shape
    dct_slice = dct.dct(lateral_slice)
    tubes = [dct_slice[i, :, 0] for i in range(l)]
    h_tubes = []
    for tube in tubes:
        h_tubes.append(torch.exp(tube) / torch.sum(torch.exp(tube)))
    res_slice = torch.stack(h_tubes, dim=0).reshape(l, m, n)
    idct_a = res_slice#dct.idct(res_slice)
    return torch.sum(idct_a, dim=0)                                                                               

def torch_apply(func, x):
    x = func(torch.transpose(x, 0, 2))
    return torch.transpose(x, 0, 2)

def make_weights(shape, device='cpu', scale=0.01):
    w = torch.randn(shape[0], 10, shape[1]) * scale
    b = torch.randn(shape[0], 10, shape[2]) * scale
    dct_w = torch_apply(dct.dct, w).to(device)
    dct_b = torch_apply(dct.dct, b).to(device)
    return dct_w, dct_b

def to_categorical(y, num_classes):
    categorical = torch.empty(len(y), num_classes)
    for i in range(len(y)):
        categorical[i, :] = torch.eye(num_classes, num_classes)[y[i]]
    return categorical

def torch_shift(A):
    x = A.squeeze()
    x = torch.transpose(x, 0, 2)
    x = torch.transpose(x, 0, 1)
    return x






