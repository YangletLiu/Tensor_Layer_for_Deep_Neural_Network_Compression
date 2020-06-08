#!/usr/bin/env python
# coding: utf-8

# # main function for decomposition
# ### Author: Yiming Fang

# In[1]:


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

from nets import *
from decomp import *


# In[2]:


# load data
def load_mnist():
    print('==> Loading data..')
    transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)
    return trainloader, testloader

def load_cifar10():
    print('==> Loading data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)
    
    return trainloader, testloader


# In[3]:


# build model
def build(model, decomp='cp'):
    print('==> Building model..')
    tl.set_backend('pytorch')
    full_net = model
    # print(full_net)
    full_net = full_net.to(device)
    torch.save(full_net, 'model')
    if decomp:
        decompose(decomp)
    net = torch.load("model").cuda()
    print(net)
    print('==> Done')
    return net
    
# training
def train(epoch, train_acc, model):
    print('\nEpoch: ', epoch)
    model.train()
    criterion = nn.CrossEntropyLoss()
    train_loss = 0
    correct = 0
    total = 0
    print('|', end='')
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if batch_idx % 10 == 0:
            print('=', end='')
    print('|', 'Accuracy:', 100. * correct / total,'% ', correct, '/', total)
    train_acc.append(correct / total)

# testing
def test(test_acc, best_acc, model):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        print('|', end='')
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if batch_idx % 10 == 0:
                print('=', end='')
    acc = 100. * correct / total
    print('|', 'Accuracy:', acc, '% ', correct, '/', total)
    test_acc.append(correct / total) 
    return max(acc, best_acc)

# decompose
def decompose(decomp):
    model = torch.load("model").cuda()
    model.eval()
    model.cpu()
    for i, key in enumerate(model.features._modules.keys()):
        if i >= len(model.features._modules.keys()) - 2:
            break
        conv_layer = model.features._modules[key]
        if isinstance(conv_layer, torch.nn.modules.conv.Conv2d):
            if decomp == 'cp':
                rank = max(conv_layer.weight.data.numpy().shape) // 5
                model.features._modules[key] = cp_decomposition_conv_layer(conv_layer, rank)
            if decomp == 'tucker':
                ranks = np.array(conv_layer.weight.data.numpy().shape) // 10
                model.features._modules[key] = tucker_decomposition_conv_layer(conv_layer, ranks)
        torch.save(model, 'model')
    return model

# Run functions
def run_train(i, model):
    train_acc = []
    test_acc = []
    best_acc = 0
    for epoch in range(i):
        train(epoch, train_acc, model)
        best_acc = test(test_acc, best_acc, model)
        scheduler.step()
        print('Current learning rate: ', scheduler.get_lr()[0])
    print('Best training accuracy overall: ', best_acc)
    return train_acc, test_acc, best_acc

def run_test(model):  
    test_acc = []
    best_acc = 0
    best_acc = test(test_acc, best_acc, model)
    print('Testing accuracy: ', best_acc)
    return test_acc, best_acc


# In[4]:


# main function
def main(): 
    global rate, trainloader, testloader, device, optimizer, scheduler
    
    # choose an appropriate learning rate
    rate = 0.1
    
    # choose dataset from (MNIST, CIFAR10, ImageNet)
    trainloader, testloader = load_mnist()
    
    # check GPU availability
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # choose model from (Net, VGG16, ...)
    model = VGG('VGG19')
    
    # choose decomposition algorithm from (CP, Tucker, TT, HT)
    net = build(model, 'tucker')
    optimizer = optim.SGD(net.parameters(), lr=rate, momentum=0.9, weight_decay=5e-4)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.8)
    
    train_acc, test_acc, _ = run_train(100, net)
    test_acc, best_acc = run_test(net)
    return best_acc


# In[5]:


if __name__ == '__main__':
    main()

