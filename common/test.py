import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import StepLR

import numpy as np
import time

def test(test_acc, model, testloader):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    s = time.time()
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
    e = time.time()
    acc = 100. * correct / total
    print('|', 'Accuracy:', acc, '% ', correct, '/', total)
    print('The inference time is', e - s, 'seconds')
    test_acc.append(correct / total)
    return test_acc, e - s