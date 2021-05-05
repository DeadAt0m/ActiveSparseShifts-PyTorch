import torch
from  torchshifts import Shift2d
from  torchshifts.quantized.modules import Shift2d as QShift2d
from  torchshifts.functional import shift2d_func

if __name__ == '__main__':
    d = torch.device('cpu')
    channels = 16
    ia = torch.rand(512,channels, 64, 64, requires_grad=True).to(d)
    ib = ia.detach().clone().to(d)
    ib.requires_grad = True
    ta = 10*torch.rand(512,channels,62,62).to(d)
    tb = ta.detach().clone()
    args = {'kernel_size':3, 'stride':1, 'padding':(0,0)}
#     args = None
    a = Shift2d(channels, init_shift=1, sparsity_term=0., active_flag=False,emulate_dw=args,
                init_thumb_rule=2).to(d)
    b = Shift2d(channels, init_shift=1, sparsity_term=0., active_flag=True,emulate_dw=args,
                init_thumb_rule=2).to(d)
    a.weight = b.weight
    la = torch.nn.MSELoss().to(d)
    lb = torch.nn.MSELoss().to(d)
    oa,_ = a(ia)
    ob,_ = b(ib)
    jac = la(oa, ta)
    jbc = lb(ob, tb)
    jac.backward()
    jbc.backward()
    assert a.weight.grad is not None, 'FAIL' 
    assert b.weight.grad is not None, 'FAIL' 
    print(a.weight.grad)
    import random
    c = random.randrange(channels)
    iq = torch.quantize_per_tensor(ia, 1/255.,0, torch.quint8)
    aq = QShift2d.from_float(a)
    oq = aq(iq)
    print('Channel:', c)
    print('Weights:', a.weight[c])
    print('Forward pass (input, output(active=False), quantized, output(active=True)):', ia[0,c], oa[0,c], oq[0,c], ob[0,c])
    ####  Test interpolation
    i,j = 3,3
    i_lb, j_lb = 1,1
    import math
    w1,w2 = a.weight[c]
    dw = lambda k: k.item() - math.floor(k) if k>0 else k.item() - math.ceil(k)
    iw = lambda k: math.floor(k) if k>0 else math.ceil(k)
    dw1 = dw(w1)
    dw2 = dw(w2)
    si = iw(w1)
    sj = iw(w2)
    a00 = ia[0,c,i-si, j-sj].item()
    a10 = ia[0,c,i+1-si, j-sj].item()
    a01 = ia[0,c,i-si,j+1-sj].item()
    a11 = ia[0,c,i+1-si,j+1-sj].item()
    def interp1D(v1, v2, x):
        return v1*(1 - x) + v2*x
    def interp2D(v1, v2, v3, v4, x, y):
        return interp1D(interp1D(v1, v2, x), interp1D(v3, v4, x), y)
    from math import floor
    print(abs(float(interp2D(a00,a10,a01,a11,dw1,dw2)) -  float(ob[0,c,i-i_lb,j-j_lb].item())))
    #test grad
    print('Backward pass (shape, SSL grad, Active grad):', ia.grad.shape, ia.grad[0,c], ib.grad[0,c])
    #CUDA
    d = torch.device('cuda:0')
    iac = ia.detach().clone().to(d)
    ibc = ib.detach().clone().to(d)
    iac.requires_grad = True
    ibc.requires_grad = True
    tac = ta.detach().clone().to(d)
    tbc = tb.detach().clone().to(d)
    from copy import deepcopy as dp
    ac = dp(a).to(d)
    bc = dp(b).to(d)
    ac.weight =bc.weight
    lac = torch.nn.MSELoss().to(d)
    lbc = torch.nn.MSELoss().to(d)
    oac,_ = ac(iac)
    obc,_ = bc(ibc)
    jac = lac(oac, tac)
    jbc = lbc(obc, tbc)
    jac.backward()
    jbc.backward()
    print('CUDAvsCPU difference: Forward pass SSL:', abs(oac.cpu()[0,c] - oa[0,c]))
    print('CUDAvsCPU difference: Forward pass Active:', abs(obc.cpu()[0,c] - ob[0,c]))
    print('CUDAvsCPU difference: Backward pass SSL:', abs(iac.grad.cpu()[0,c] - ia.grad[0,c]))
    print('CUDAvsCPU difference: Backward pass Active:', abs(ibc.grad.cpu()[0,c] - ib.grad[0,c]))