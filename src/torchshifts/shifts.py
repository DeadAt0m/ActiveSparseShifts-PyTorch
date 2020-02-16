from torch import nn
from torch.autograd import Function
import torch

#  import the CPU shifts
SHIFTS_CPU_AVAILABLE = True
try:
    from torchshifts.cpu import shift2d_cpu, shift2d_backward_cpu
except ImportError:
    SHIFTS_CPU_AVAILABLE = False

# try to import the CUDA shifts
SHIFTS_GPU_AVAILABLE = True
try:
    from torchshifts.cuda import shift2d_gpu, shift2d_backward_gpu
except ImportError:
    SHIFTS_GPU_AVAILABLE = False
    
    
class shift2d_func(Function):
    @staticmethod
    def forward(ctx, input, weight, padding_mode):
        assert type(padding_mode) == int, f'shift2d_func() expected int padding_mode'
        assert padding_mode in [0,1], f'shift2d_func() expected padding_mode can be 0 - zeros or 1 - border'
        assert len(input.shape) == 4, f'shift2d_func(): expected 4D tensor of shape (N,C,H,W) as input, but it is shape is {input.shape}'
        assert len(weight.shape) == 2, f'shift2d_func(): expected 2D tensor of shape (C,2) as weight, but it is shape is {weight.shape}'
        
        assert input.shape[1] == weight.shape[0],  f'shift2d_func(): expected that input and weight have equal number of channels, but input have {input.shape[1]} and weight have {weight.shape[0]} channels.'
        assert input.device == weight.device, f'shift2d_func(): expected input and weights to be on same device, but input is  on {input.device} and weights is on {weight.device}'

        if input.is_cuda and not SHIFTS_GPU_AVAILABLE:
            raise Exception('shifts on CUDA device is asked, but it seems that it is not available. Please install it')
        if not input.is_cuda and not SHIFTS_CPU_AVAILABLE:
            raise Exception('shifts on CPU is not available. Please install it.')

        ctx.padding_mode = padding_mode
        ctx.save_for_backward(input, weight)
        if input.is_cuda:
            output = shift2d_gpu(input, weight)
        else:
            output = shift2d_cpu(input, weight)
        return output

    @staticmethod
    def backward(ctx, grad_ouput):
        input, weight = ctx.saved_tensors
        if input.is_cuda:
            grad_input, grad_weight = shift2d_backward_gpu(input, weight, input, ctx.padding_mode)
        else:
            grad_input, grad_weight = shift2d_backward_cpu(input, weight, input, ctx.padding_mode)
        return grad_input, grad_weight, None

class Shift2D(nn.Module):
    def __init__(self, in_channels, padding='zeros',
                 init_stride = 1,
                 sparsity_term=5e-4):
        super(Shift2D, self).__init__()
        self.padding_dict = {'zeros':0, 'border':1}
        assert padding.lower() in self.padding_dict.keys(), f'incorrect padding option: {padding}'
        self.padding = self.padding_dict[padding]
        self.sparsity_term = sparsity_term
        self.weight = nn.Parameter(torch.Tensor(in_channels,2))
        self.reset_parameters(init_stride)
        
    def reset_parameters(self, init_stride):
        self.weight.data.uniform_(-abs(init_stride), abs(init_stride))
    
    def __compute_weight_loss(self):
        return self.sparsity_term * torch.sum(torch.abs(self.weight))
                
    def forward(self, input):
        if bool(self.sparsity_term):
            return shift2d_func.apply(input, self.weight, self.padding), self.__compute_weight_loss()
        return shift2d_func.apply(input, self.weight, self.padding)
