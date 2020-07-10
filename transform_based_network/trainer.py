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


def train_step_transform(epoch, train_acc, model, trainloader, optimizer):  
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    
    print('\nEpoch: ', epoch)
    print('|', end='')
    for batch_idx, (inputs, labels) in enumerate(trainloader):   
        inputs = inputs.to(device)
        if not inputs.shape[0] == 100:
            break

        labels = labels.to(device)
        outputs = model(inputs) 
        outputs = torch.transpose(scalar_tubal_func(outputs), 0, 1)
        
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        # print(loss)
        if np.isnan(loss.item()):
            print('Training terminated due to instability')
            break
        loss.backward()
        optimizer.step()
        train_loss += loss.item()  
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        if batch_idx % 10 == 0:
            print('=', end='')
    print('|', 'Accuracy:', correct / total, 'Loss:', train_loss / total)
    train_acc.append(correct / total)
    return train_acc

def test(test_acc, model, testloader):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    s = time.time()
    with torch.no_grad():
        print('|', end='')
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs = inputs.to(device)
            if not inputs.shape[0] == 100:
                break
            targets = targets.to(device)
            outputs = model(inputs) 
            outputs = torch.transpose(scalar_tubal_func(outputs), 0, 1)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if batch_idx % 10 == 0:
                print('=', end='')
    e = time.time() 
    print('|', ' Test accuracy:', correct / total, 'Test loss:', test_loss / total)
    print('The inference time is', e - s, 'seconds')
    test_acc.append(correct / total)
    return test_acc, e - s
    
def train_transform(i, model, trainloader, testloader, optimizer):
    train_acc, test_acc = [], []
    scheduler = StepLR(optimizer, step_size=1, gamma=0.95)
    
    for epoch in range(i):
        s = time.time()
        train_acc = train_step_transform(epoch, train_acc, model, trainloader, optimizer)
        e = time.time()
        test_acc, _ = test(test_acc, model, testloader)
        scheduler.step()
        
        print('This epoch took', e - s, 'seconds to train')
        print('Current learning rate:', scheduler.get_last_lr()[0])
    print('Best training accuracy overall:', max(test_acc))
    return train_acc, test_acc