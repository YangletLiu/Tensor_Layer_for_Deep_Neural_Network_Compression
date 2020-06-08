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


def cp_decomposition_conv_layer(layer, rank):
    l, f, v, h = parafac(layer.weight.data, rank=rank)[1]
    
    pointwise_s_to_r_layer = torch.nn.Conv2d(
            in_channels=f.shape[0], 
            out_channels=f.shape[1], 
            kernel_size=1, 
            stride=1, 
            padding=0, 
            dilation=layer.dilation, 
            bias=False)

    depthwise_vertical_layer = torch.nn.Conv2d(
            in_channels=v.shape[1], 
            out_channels=v.shape[1], 
            kernel_size=(v.shape[0], 1),
            stride=1, padding=(layer.padding[0], 0), 
            dilation=layer.dilation,
            groups=v.shape[1], 
            bias=False)

    depthwise_horizontal_layer = torch.nn.Conv2d(
            in_channels=h.shape[1], 
            out_channels=h.shape[1], 
            kernel_size=(1, h.shape[0]), 
            stride=layer.stride,
            padding=(0, layer.padding[0]), 
            dilation=layer.dilation, 
            groups=h.shape[1], 
            bias=False)

    pointwise_r_to_t_layer = torch.nn.Conv2d(
            in_channels=l.shape[1], 
            out_channels=l.shape[0], 
            kernel_size=1, 
            stride=1,
            padding=0, 
            dilation=layer.dilation, 
            bias=True)
    
    pointwise_r_to_t_layer.bias.data = layer.bias.data
    depthwise_horizontal_layer.weight.data = torch.transpose(h, 1, 0).unsqueeze(1).unsqueeze(1)
    depthwise_vertical_layer.weight.data = torch.transpose(v, 1, 0).unsqueeze(1).unsqueeze(-1)
    pointwise_s_to_r_layer.weight.data = torch.transpose(f, 1, 0).unsqueeze(-1).unsqueeze(-1)
    pointwise_r_to_t_layer.weight.data = l.unsqueeze(-1).unsqueeze(-1)

    new_layers = [pointwise_s_to_r_layer, depthwise_vertical_layer, 
                  depthwise_horizontal_layer, pointwise_r_to_t_layer]
    
    return nn.Sequential(*new_layers)

def tucker_decomposition_conv_layer(layer, ranks):
    core, [last, first] = partial_tucker(layer.weight.data, modes=[0, 1], ranks=ranks, init='svd')

    # A pointwise convolution that reduces the channels from S to R3
    first_layer = torch.nn.Conv2d(in_channels=first.shape[0], 
                out_channels=first.shape[1], kernel_size=1,
                stride=1, padding=0, dilation=layer.dilation, bias=False)

    # A regular 2D convolution layer with R3 input channels 
    # and R3 output channels
    core_layer = torch.nn.Conv2d(in_channels=core.shape[1], 
                out_channels=core.shape[0], kernel_size=layer.kernel_size,
                stride=layer.stride, padding=layer.padding, dilation=layer.dilation, bias=False)

    # A pointwise convolution that increases the channels from R4 to T
    last_layer = torch.nn.Conv2d(in_channels=last.shape[1], \
                out_channels=last.shape[0], kernel_size=1, stride=1,
                padding=0, dilation=layer.dilation, bias=True)

    last_layer.bias.data = layer.bias.data

    first_layer.weight.data = torch.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1)
    last_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)
    core_layer.weight.data = core

    new_layers = [first_layer, core_layer, last_layer]
    return nn.Sequential(*new_layers)

def tt_decomposition_conv_layer(layer, ranks):
    cores = matrix_product_state(layer.weight.data, rank)
    core_layers = []
    
    for core in range(1, len(cores) - 1):
        core_layer = torch.nn.Conv2d(in_channels=core.shape[1], 
                out_channels=core.shape[0], kernel_size=layer.kernel_size,
                stride=layer.stride, padding=layer.padding, dilation=layer.dilation, bias=False)
        core_layer.weight.data = core

    return nn.Sequential(*cores)