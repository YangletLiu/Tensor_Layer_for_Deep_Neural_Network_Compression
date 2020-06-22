import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import StepLR

import torchvision
import torchvision.transforms as transforms
from torchvision import models

import tensorly as tl
import tensorly
from itertools import chain
from tensorly.decomposition import parafac, partial_tucker

import os
import matplotlib.pyplot as plt
import numpy as np
import time

import sys
sys.path.append('..')
from common import *
from decomposition import *
from nets import *


# main function
def run_all(dataset, model, decomp=None, i=100, rate=0.05): 
    
    # choose dataset from (MNIST, CIFAR10, ImageNet)
    if dataset == 'mnist':
        trainloader, testloader = load_mnist()
    if dataset == 'cifar10':
        trainloader, testloader = load_cifar10()
    if dataset == 'cifar100':
        trainloader, testloader = load_cifar100()

    # choose decomposition algorithm from (CP, Tucker, TT)
    net = build(model, decomp)
    optimizer = optim.SGD(net.parameters(), lr=rate, momentum=0.9, weight_decay=5e-4)
    train_acc, test_acc = train(i, net, trainloader, testloader, optimizer)
    _, inf_time = test([], net, testloader)
    
    if not decomp:
        decomp = 'full'
    filename = dataset + '_' + decomp
    
    torch.save(net, 'models/' + filename)
    path = 'curves/'
    if not os.path.exists(path):
        os.mkdir(path)
        
    np.save(path + filename + '_train', train_acc)
    np.save(path + filename + '_test', test_acc)
