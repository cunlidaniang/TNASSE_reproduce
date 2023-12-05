import numpy as np
import torch
import copy

from nasbench import api
from Model import Network

INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'

def CodeToSpec(code):
    if not isinstance(code, tuple):
        raise ValueError('code must be tuple')
    if len(code) != 3:
        raise ValueError('code must be 3')
    l1,l2,l3=code
    if len(l1) != 5:
        raise ValueError('len(l1) must be 5')
    if len(l2) != 21:
        raise ValueError('len(l2) must be 21')
    if len(l3) != 5:
        raise ValueError('len(l3) must be 5')
    cnt=0
    matrix=np.zeros((7,7))
    for i in range(0,7):
        for j in range(i+1,7):
            matrix[i][j]=l2[cnt]
            cnt+=1
    opts=[]
    opts.append(INPUT)
    for i in range(0,5):
        if l1[i] == 1:
            opts.append(CONV1X1)
        elif l1[i] == 2:
            opts.append(CONV3X3)
        elif l1[i] == 3:
            opts.append(MAXPOOL3X3)
    opts.append(OUTPUT)
    spec=api.ModelSpec(matrix=matrix,ops=opts)
    return spec

def SpecToNet(spec):
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return Network(spec, device)

def cal_score(ori, noi, rn, n_conv, channel):
    errs=[]
    for i in range(len(ori)):
        error = ori[i] - noi[i]
        errs.append(np.sum(np.square(error))/error.size)
    epsilon=1e-10
    theta = 0
    eta = np.log(epsilon+np.sum(errs))
    gamma = channel
    rho = n_conv/rn
    if eta>theta:
        Psi = np.log((gamma*rho)/eta)
    else:
        Psi = 0
    return Psi

K = []
rn = 0
n_conv=0
channel = 0

def NetToScore(network, x):
    global K,rn,n_conv,channel
    network1 = copy.deepcopy(network)
    network2 = copy.deepcopy(network)
    network1 = network1.cuda()
    network2 = network2.cuda()
    def counting_forward_hook(module, inp, out):
        global K,rn,channel
        if not module.visited_backwards:
            return
        if isinstance(inp, tuple):
             inp = inp[0]
        arr = inp.detach().cpu().numpy()
        K.append(arr) 
        rn = rn + 1
        channel+=arr.shape[1]

    def counting_backward_hook(module, inp, out):
        module.visited_backwards = True

    def counting_forward_hook_conv(module, inp, out):
        global K,n_conv,channel
        if not module.visited_backwards_conv:
            return
        if isinstance(inp, tuple):
            inp = inp[0]
        arr = inp.detach().cpu().numpy()               
        n_conv = n_conv + 1
        channel+=arr.shape[1]
        
    def counting_backward_hook_conv(module, inp, out):
        module.visited_backwards_conv = True
        
    for name, module in network1.named_modules():
        if 'Pool' in str(type(module)):
            module.register_forward_hook(counting_forward_hook)
            module.register_backward_hook(counting_backward_hook)
            module.visited_backwards = False
        if 'Conv' in str(type(module)):
            module.register_forward_hook(counting_forward_hook_conv)
            module.register_backward_hook(counting_backward_hook_conv)
            module.visited_backwards_conv = False
    for name, module in network2.named_modules():
        if 'Pool' in str(type(module)):
            module.register_forward_hook(counting_forward_hook)
            module.register_backward_hook(counting_backward_hook)
            module.visited_backwards = False
        if 'Conv' in str(type(module)):
            module.register_forward_hook(counting_forward_hook_conv)
            module.register_backward_hook(counting_backward_hook_conv)
            module.visited_backwards_conv=False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x_origin = torch.clone(x)
    x_origin = x_origin.to(device)
    x_noise = torch.clone(x)
    x_noise = x_noise.to(device)
    noise = (x.new(x.size()).normal_(0,0.05)).to(device)
    x_noise = x_noise + noise
    network1.zero_grad()
    x_origin.requires_grad_(True)
    y = network1(x_origin, get_ints=False)
    y.backward(torch.ones_like(y))
    y = network1(x_origin, get_ints=False)
    KK=copy.deepcopy(K)
    K = []
    rn = 0
    n_conv=0
    channel = 0
    network2.zero_grad()
    x_noise.requires_grad_(True)
    y = network2(x_noise, get_ints=False)
    y.backward(torch.ones_like(y))
    y = network2(x_noise, get_ints=False)
    KKK=copy.deepcopy(K)
    return cal_score(KK,KKK,rn,n_conv,channel)
            
