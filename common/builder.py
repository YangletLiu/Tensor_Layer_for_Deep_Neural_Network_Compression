import torch
import tensorly as tl

def build(model, decomp='cp'):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('==> Building model..')
    tl.set_backend('pytorch')
    full_net = model
    full_net = full_net.to(device)
    torch.save(full_net, 'models/model')
    if decomp:
        decompose_all(decomp)
    if device == 'cuda:0':
        net = torch.load("models/model").cuda()
    else:
        net = torch.load("models/model")
    print(net)
    print('==> Done')
    return net