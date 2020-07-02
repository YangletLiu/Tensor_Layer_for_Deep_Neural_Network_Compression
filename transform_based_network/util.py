import sys
sys.path.append('../')
from common import *

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
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return torch.Tensor(toeplitz(A) + hankel(A)).to(device)

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
    # assert(A.shape[0] == B.shape[0] and A.shape[2] == B.shape[1])
    dct_C = torch.zeros(A.shape[0], A.shape[1], B.shape[2])
    for k in range(A.shape[0]):
        dct_C[..., k] = torch.mm(dct.dct(A)[..., k], dct.dct(B)[..., k])
    return dct.idct(dct_C)

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
    for slice in lateral_slices:
        h_slice.append(h_func_dct(slice))
    pro_matrix = torch.stack(h_slice, dim=2)
    return pro_matrix.reshape(m, n)

def raw_img(img, batch_size, n):
    img_raw = img.reshape(batch_size, n * n)
    single_img = torch.split(img_raw, split_size_or_sections=1, dim=0)
    single_img_T = [torch.transpose(i.reshape(n, n, 1), 0, 1) for i in single_img]
    ultra_img = torch.cat(single_img_T, dim=2)
    return ultra_img








