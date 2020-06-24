import torch
import torch.nn as nn
import torchvision
from torchvision import models
import tensorly as tl
import tensorly
from itertools import chain

from .conv_layer import *
from .fc_layer import *

# decompose all layers in a network
def decompose_conv(decomp):
    model = torch.load("models/model").cuda()
    model.eval()
    model.cpu()
    for i, key in enumerate(model.features._modules.keys()):
        if i >= len(model.features._modules.keys()) - 2:
            break
        conv_layer = model.features._modules[key]
        if isinstance(conv_layer, torch.nn.modules.conv.Conv2d):
            rank = max(conv_layer.weight.data.numpy().shape) // 3
            if decomp == 'cp':
                model.features._modules[key] = cp_decomposition_conv_layer(conv_layer, rank)
            if decomp == 'tucker': 
                ranks = [int(np.ceil(conv_layer.weight.data.numpy().shape[0] / 3)), 
                         int(np.ceil(conv_layer.weight.data.numpy().shape[1] / 3))]
                model.features._modules[key] = tucker_decomposition_conv_layer(conv_layer, ranks)
            if decomp == 'tt':
                model.features._modules[key] = tt_decomposition_conv_layer(conv_layer, rank)
        torch.save(model, 'models/model')
                
def decompose_fc(decomp):
    model = torch.load("models/model").cuda()
    model.eval()
    model.cpu()
    for i, key in enumerate(model.classifier._modules.keys()):
        linear_layer = model.classifier._modules[key]
        if isinstance(linear_layer, torch.nn.modules.linear.Linear):
            rank = min(linear_layer.weight.data.numpy().shape) // 2
            if decomp == 'tucker':
                model.classifier._modules[key] = tucker_decomposition_fc_layer(linear_layer, rank)
            else:
                model.classifier._modules[key] = decomposition_fc_layer(linear_layer, rank)
        torch.save(model, 'models/model')
    return model