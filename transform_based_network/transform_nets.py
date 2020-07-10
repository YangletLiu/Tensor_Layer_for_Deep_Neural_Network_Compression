import sys
sys.path.append('../')
from common import *
from .transform_layer import *

import torch
import torch_dct as dct
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import StepLR


class Transform_Net(nn.Module):
    def __init__(self, batch_size):
        super(Transform_Net, self).__init__()
        self.features = nn.Sequential(
            Transform_Layer(28, 28, batch_size, 28),
            nn.ReLU(inplace=True),
            Transform_Layer(28, 28, batch_size, 28),
            nn.ReLU(inplace=True),
            Transform_Layer(28, 28, batch_size, 10),
        )

    def forward(self, x):
        x = raw_img(x, x.size(0), 28)
        x = self.features(x)        
        return x / 1e5
    

class Conv_Transform_Net(nn.Module):
    def __init__(self, batch_size):
        super(T_Net, self).__init__()
        self.first = nn.Sequential(
            Transform_Layer(28, 28, batch_size, 28),
            nn.ReLU(inplace=True),
        ) 
        self.intermediate = nn.Sequential(
            nn.Conv2d(28, 28, kernel_size=3, padding=1),
            nn.Conv2d(28, 28, kernel_size=3, padding=1),
            nn.Conv2d(28, 28, kernel_size=1, padding=0),
        )
        self.last = Transform_Layer(28, 28, batch_size, 10)

    def forward(self, x):
        x = raw_img(x, x.size(0), 28)
        x = self.first(x)
        
        x = torch.transpose(x, 0, 2)
        x = torch.transpose(x, 1, 2)
        x = x.reshape(100, 28, 4, 7)
        
        x = self.intermediate(x)
        
        x = x.reshape(100, 28, 28)
        x = torch.transpose(x, 0, 2)
        x = torch.transpose(x, 0, 1)
        x = self.last(x)
        
        return x / 5e2
    
    
class Conv_Transform_Net_CIFAR(nn.Module):
    def __init__(self, batch_size):
        super(T_Net_CIFAR, self).__init__()
        self.batch_size = batch_size
        self.first = nn.Sequential(
            Transform_Layer(96, 32, batch_size, 32),
            nn.ReLU(inplace=True),
            Transform_Layer(96, 32, batch_size, 32),
            nn.ReLU(inplace=True),
        ) 
        
        self.intermediate = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
        )
        self.last = Transform_Layer(96, 32, batch_size, 10)
        
    def forward(self, x):
        x = torch.reshape(x, (100, 96, 32))
        x = torch.transpose(x, 0, 2)
        x = torch.transpose(x, 0, 1)
        x = self.first(x)
        
        x = torch.transpose(x, 0, 2)
        x = torch.transpose(x, 1, 2)
        x = x.reshape(self.batch_size, 96, 4, 8)
        x = self.intermediate(x)

        x = x.reshape(self.batch_size, 96, 32)
        x = torch.transpose(x, 0, 2)
        x = torch.transpose(x, 0, 1)
        x = self.last(x)
        
        return x