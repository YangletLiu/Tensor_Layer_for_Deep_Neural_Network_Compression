#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np


# In[76]:


def unfold(A):
    return torch.tensor(bcirc(A)[:, :A.shape[1]])

def fold(A, shape):
    return torch.tensor(np.reshape(A, shape))

def bcirc(A):
    s = A.shape
    bcirc_A = np.zeros((s[0] * s[2], s[1] * s[2]))
    A_matriced = np.reshape(np.transpose(A, [0, 1, 2]), (s[0] * s[2], s[1]))
    for k in range(s[2]):
        bcirc_A[:, k * s[1] : (k + 1) * s[1]] = np.roll(A_matriced, k * s[0], axis=0)
    return torch.tensor(bcirc_A)

def t_product(A, B):
    fft_A = np.fft.fft(A, axis=-1)
    fft_B = np.fft.fft(B, axis=-1)
    fft_C = np.zeros((A.shape[0], B.shape[1], A.shape[2]), dtype=complex)
    for k in range(A.shape[2]):
        fft_C[..., k] = fft_A[..., k] @ fft_B[..., k]
    return torch.tensor(np.real(np.fft.ifft(fft_C, axis=-1)))

def t_product_v2(A, B):
    shape = [A.shape[0], B.shape[1], A.shape[2]]
    return torch.tensor(fold(bcirc(A) @ unfold(B), shape))

def loss(W, A, C):
    return np.linalg.norm(C - t_product(W, A)) ** 2 / 2


# In[81]:


A = torch.rand(2, 3, 4)
B = torch.rand(3, 5, 4)

print(t_product(A, B))

