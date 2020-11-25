import torch
import torch_dct as dct
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn.init as init
from torch.optim.lr_scheduler import StepLR

import matplotlib.pyplot as plt
import time
import pkbar
import math

import sys
sys.path.append('../')
from common import *
from transform_based_network import *


class tNN(nn.Module):
    def __init__(self):
        super(tNN, self).__init__()
        W, B = [], []
        self.num_layers = 4
        for i in range(self.num_layers):
            W.append(nn.Parameter(torch.Tensor(28, 28, 28)))
            B.append(nn.Parameter(torch.Tensor(28, 28, 1)))
        self.W = nn.ParameterList(W)
        self.B = nn.ParameterList(B)
        self.reset_parameters()

    def forward(self, x):
        for i in range(self.num_layers):
            x = torch.add(t_product(self.W[i], x), self.B[i])
            x = F.relu(x)
        return x

    def reset_parameters(self):
        for i in range(self.num_layers):
            init.kaiming_uniform_(self.W[i], a=math.sqrt(5))
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.W[i])
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.B[i], -bound, bound)