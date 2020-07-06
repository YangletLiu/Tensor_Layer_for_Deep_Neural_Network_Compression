import sys
sys.path.append('../')
from common import *

import torch
import torch_dct as dct
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import StepLR
import sys
sys.path.append('..')
from transform_based_network import *
from common import *

class Transform_Layer(nn.Module):
    def __init__(self, n, size_in, m, size_out):
        super().__init__()
        self.size_in = size_in
        self.size_out = size_out
        weights = torch.randn(n, size_out, size_in)
        bias = torch.randn(1, size_out, m)
        self.weights = nn.Parameter(weights, requires_grad=True)
        self.bias = nn.Parameter(bias, requires_grad=True)
        
    def forward(self, x):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        return torch.add(t_product(self.weights, x).to(device), self.bias)