import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_dct as dct

import sys
sys.path.append('../')
from common import *
from transform_based_network import *

from torch.multiprocessing import Pool, Queue, Process, set_start_method
import torch.multiprocessing as torch_mp
import multiprocessing as mp

if __name__ == '__main__':
    tmp = torch_mp.get_context('spawn')

    trainloader, testloader = load_mnist()
    model = Transform_Net(100)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    model.share_memory()

    processes = []
    num_cores = torch_mp.cpu_count()
    for i in range(num_cores):
        # q = Queue()
        p = tmp.Process(target=train_transform, args=(1, model, trainloader, testloader, optimizer))
        p.start()
        # print(q.get())
        processes.append(p)

    for p in processes: 
        p.join()