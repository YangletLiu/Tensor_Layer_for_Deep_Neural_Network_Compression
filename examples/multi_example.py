#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch_dct as dct
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import StepLR

import matplotlib.pyplot as plt
plt.style.use(['science','no-latex', 'notebook'])

import time
import sys
import PIL
sys.path.append('../')
from common import *
from transform_based_network import *


# In[4]:


trainloader, testloader = load_cifar10()
model = Conv_Transform_Net_CIFAR(100)
print(model)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
train_acc, test_acc = train_transform(1, model, trainloader, testloader, optimizer)

# In[ ]:


plt.figure()
plt.title('Convolutional Transform Net Accuracy on CIFAR10')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(train_acc, label='Train accuracy')
plt.plot(test_acc, label='Test accuracy')
plt.legend()
plt.show()

