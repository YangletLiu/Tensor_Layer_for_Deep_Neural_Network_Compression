import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import StepLR

import numpy as np
import time
from .test import *

def train_step(epoch, train_acc, model, trainloader, optimizer):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
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
    return train_acc

def train(i, model, trainloader, testloader, optimizer):
    train_acc = []
    test_acc = []
    scheduler = StepLR(optimizer, step_size=5, gamma=0.9)
    for epoch in range(i):
        s = time.time()
        train_acc = train_step(epoch, train_acc, model, trainloader, optimizer)
        test_acc, _ = test(test_acc, model, testloader)
        scheduler.step()
        e = time.time()
        print('This epoch took', e - s, 'seconds to train')
        print('Current learning rate: ', scheduler.get_last_lr()[0])
    print('Best training accuracy overall: ', max(test_acc))
    return train_acc, test_acc