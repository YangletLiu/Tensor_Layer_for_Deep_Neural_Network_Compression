import torch
import tensorly as tl
import os
import sys
sys.path.append('..')
from decomposition import *
             

def build(model, decomp='cp'):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('==> Building model..')
    tl.set_backend('pytorch')
    full_net = model
    full_net = full_net.to(device)
 
    path = 'models/'
    if not os.path.exists(path):
        os.mkdir(path)
    torch.save(full_net, path + 'model')
    if decomp:
        decompose_conv(decomp)
        decompose_fc(decomp)
    if device == 'cuda:0':
        net = torch.load(path + "model").cuda()
    else:
        net = torch.load(path + "model")
    print(net)
    print('==> Done')
    return net