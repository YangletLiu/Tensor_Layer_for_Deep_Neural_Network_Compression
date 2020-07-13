import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_dct as dct

import time
from torch.multiprocessing import Pool, Queue, Process, set_start_method
import torch.multiprocessing as torch_mp
import multiprocessing as mp_

import sys
sys.path.append('../')
from common import *
from transform_based_network import *


def t_product_slice(A, B, C, i):
    C[i, ...] = torch.mm(A[i, ...], B[i, ...])

def t_product_multiprocess(A, B):
    tmp = torch_mp.get_context('spawn')
    
    assert(A.shape[0] == B.shape[0] and A.shape[2] == B.shape[1])
    dct_A = torch.transpose(dct.dct(torch.transpose(A, 0, 2)), 0, 2)
    dct_B = torch.transpose(dct.dct(torch.transpose(B, 0, 2)), 0, 2)
    dct_C = torch.zeros(A.shape[0], A.shape[1], B.shape[2])
    
    #dct_A.share_memory_()
    #dct_B.share_memory_()
    #dct_C.share_memory_()
    
    processes = []
    # num_cores = torch_mp.cpu_count()
    for i in range(dct_C.shape[0]):
        p = tmp.Process(target=t_product_slice, args=(dct_A, dct_B, dct_C, i))
        p.start()
        processes.append(p)
    for p in processes: 
        p.join()
     
    C = torch.transpose(dct.idct(torch.transpose(dct_C, 0, 2)), 0, 2)
    return C

'''
if __name__ == '__main__':
    A = torch.ones(17, 1000, 1000)
    B = torch.ones(17, 1000, 1000)
    t_product_multiprocess(A, B)
'''