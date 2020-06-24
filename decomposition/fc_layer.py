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
from tensorly.decomposition import parafac, tucker, matrix_product_state

import os
import matplotlib.pyplot as plt
import numpy as np
import time


def decomposition_fc_layer(layer, rank):
    l, r = matrix_product_state(layer.weight.data, rank=rank)
    l, r = l.squeeze(), r.squeeze()
    
    right_layer = torch.nn.Linear(r.shape[1], r.shape[0])
    left_layer = torch.nn.Linear(l.shape[1], l.shape[0])
    
    left_layer.bias.data = layer.bias.data
    left_layer.weight.data = l
    right_layer.weight.data = r

    new_layers = [right_layer, left_layer]
    return nn.Sequential(*new_layers)


def tucker_decomposition_fc_layer(layer, rank):
    core, [l, r] = tucker(layer.weight.data, rank=rank)
    
    right_layer = torch.nn.Linear(r.shape[0], r.shape[1])
    core_layer = torch.nn.Linear(core.shape[1], core.shape[0])
    left_layer = torch.nn.Linear(l.shape[1], l.shape[0])
    
    left_layer.bias.data = layer.bias.data
    left_layer.weight.data = l
    right_layer.weight.data = r.T

    new_layers = [right_layer, core_layer, left_layer]
    return nn.Sequential(*new_layers)