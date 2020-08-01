import torch
import torch_dct as dct
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import StepLR

import matplotlib.pyplot as plt

from multiprocessing import Pool, Queue, Process, set_start_method
import multiprocessing as mp_
from joblib import Parallel, delayed

import time
import pkbar
import sys
sys.path.append('../')
from common import *
from transform_based_network import *


class T_Layer(nn.Module):
    def __init__(self, dct_w, dct_b):
        super(T_Layer, self).__init__()
        self.weights = nn.Parameter(dct_w)
        self.bias = nn.Parameter(dct_b)
        
    def forward(self, dct_x):
        x = torch.mm(self.weights, dct_x) + self.bias
        return x

    
class Frontal_Slice(nn.Module):
    def __init__(self, dct_w, dct_b):
        super(Frontal_Slice, self).__init__()
        self.device = dct_w.device
        self.dct_linear = nn.Sequential(
            T_Layer(dct_w, dct_b),
        )
        #nn.ReLU(inplace=True),
        #self.linear1 = nn.Linear(28, 28)
        #nn.ReLU(inplace=True),
        #self.classifier = nn.Linear(28, 10)
        
    def forward(self, x):
        #x = torch.transpose(x, 0, 1).to(self.device)
        x = self.dct_linear(x)
        #x = self.linear1(x)
        #x = self.classifier(x)
        #x = torch.transpose(x, 0, 1)
        return x


def train_slice(i, model, x_i, y, outputs, optimizer):
    s = time.time()
    criterion = nn.CrossEntropyLoss()
    o = torch.stack(outputs)
    o[i, ...] = outputs_grad[i]
    o = torch_apply(dct.idct, o)
    o = scalar_tubal_func(o)
    o = torch.transpose(o, 0, 1)
    
    optimizer.zero_grad()
    loss = criterion(o, y) 
    loss.backward()
    optimizer.step()
    e = time.time()
    # print(e - s)


device = 'cpu'
batch_size = 100
trainloader, testloader = load_mnist_multiprocess(batch_size)


shape = (28, 28, batch_size)
models = []
ops = []
dct_w, dct_b = make_weights(shape, device=device)
for i in range(shape[0]):
    w_i = dct_w[i, ...].clone()
    b_i = dct_b[i, ...].clone()
    
    w_i.requires_grad = True
    b_i.requires_grad = True
    
    model = Frontal_Slice(w_i, b_i)
    model.train()
    models.append(model.to(device))
    
    op = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    ops.append(op)


epochs = 10
acc_list = []
loss_list = []

global outputs_grad
for e in range(epochs):
    correct = 0
    total = 0
    losses = 0
    pbar = pkbar.Pbar(name='Epoch '+str(e), target=60000/batch_size)
    for batch_idx, (x, y) in enumerate(trainloader):   
        dct_x = torch_shift(x)
        dct_x = torch_apply(dct.dct, dct_x)

        dct_x = dct_x.to(device)
        y = y.to(device)            
        
        outputs_grad = []
        outputs = []
        
        for i in range(len(models)):
            out = models[i](dct_x[i, ...])
            outputs_grad.append(out)
            outputs.append(out.detach())
        
        #for i in range(len(models)):
        #    train_slice(i, models[i], dct_x[i, ...], y, outputs, ops[i])
            
        Parallel(n_jobs=16, prefer="threads", verbose=0)(
            delayed(train_slice)(i, models[i], dct_x[i, ...], y, outputs, ops[i]) \
            for i in range(len(models))
        )

        res = torch.empty(shape[0], 10, shape[2])
        for i in range(len(models)):
            res[i, ...] = models[i](dct_x[i, ...])
            
        res = torch_apply(dct.idct, res).to(device)
        res = scalar_tubal_func(res)
        res = torch.transpose(res, 0, 1)
        criterion = nn.CrossEntropyLoss()
        total_loss = criterion(res, y)
        
        _, predicted = torch.max(res, 1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()
        losses += total_loss
        
        pbar.update(batch_idx)
        # print(total_loss)
        # print(predicted.eq(y).sum().item() / y.size(0))
        
    loss_list.append(losses / total)
    3acc_list.append(correct / total)


'''
    tmp = torch_mp.get_context('spawn')
    for model in models:
        model.share_memory()
    processes = []

    for i in range(len(models)):
        p = tmp.Process(target=train_slice, 
            args=(i, models[i], dct_x[i, ...], y, outputs, ops[i]))
        p.start()
        processes.append(p)
    for p in processes: 
        p.join()
    '''