from torch import nn
from torch.autograd import Function
import torch

#  import the CPU shifts
SHIFTS_CPU_AVAILABLE = True
try:
    from torchshifts.cpu import shift1d_cpu, shift1d_backward_cpu, shift2d_cpu, shift2d_backward_cpu
except ImportError:
    SHIFTS_CPU_AVAILABLE = False

# try to import the CUDA shifts
SHIFTS_GPU_AVAILABLE = True
try:
    from torchshifts.cuda import shift1d_gpu, shift1d_backward_gpu, shift2d_gpu, shift2d_backward_gpu
except ImportError:
    SHIFTS_GPU_AVAILABLE = False
    
    
class shift1d_func(Function):
    @staticmethod
    def forward(ctx, input, weight, padding_mode, active_flag):
        assert type(padding_mode) == int, f'shift1d_func() expected int padding_mode'
        assert padding_mode in [0,1,2,3], f'shift1d_func() expected padding_mode can be 0 - zeros, 1 - border, 2 - reflect, 3 - symmetric'
        assert len(input.shape) == 3, f'shift1d_func(): expected 3D tensor of shape (N,C,H) as input, but it is shape is {input.shape}'
        assert len(weight.shape) == 1, f'shift1d_func(): expected 1D tensor of shape (C,) as weight, but it is shape is {weight.shape}'
        
        assert input.shape[1] == weight.shape[0],  f'shift1d_func(): expected that input and weight have equal number of channels, but input have {input.shape[1]} and weight have {weight.shape[0]} channels.'
        assert input.device == weight.device, f'shift1d_func(): expected input and weights to be on same device, but input is  on {input.device} and weights is on {weight.device}'

        if input.is_cuda and not SHIFTS_GPU_AVAILABLE:
            raise Exception('shifts on CUDA device is asked, but it seems that it is not available. Please install it')
        if not input.is_cuda and not SHIFTS_CPU_AVAILABLE:
            raise Exception('shifts on CPU is not available. Please install it.')

        ctx.padding_mode = padding_mode
        ctx.active_flag = active_flag
        ctx.save_for_backward(input, weight)
        if input.is_cuda:
            output = shift1d_gpu(input, weight, padding_mode, active_flag)
        else:
            output = shift1d_cpu(input, weight, padding_mode, active_flag)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        if input.is_cuda:
            grad_input, grad_weight = shift1d_backward_gpu(grad_output, weight, input, ctx.padding_mode, ctx.active_flag)
        else:
            grad_input, grad_weight = shift1d_backward_cpu(grad_output, weight, input, ctx.padding_mode, ctx.active_flag)
        return grad_input, grad_weight, None, None  
    

class shift2d_func(Function):
    @staticmethod
    def forward(ctx, input, weight, padding_mode, active_flag):
        assert type(padding_mode) == int, f'shift2d_func() expected int padding_mode'
        assert padding_mode in [0,1], f'shift2d_func() expected padding_mode can be 0 - zeros, 1 - border, 2 - reflect, 3 - symmetric'
        assert len(input.shape) == 4, f'shift2d_func(): expected 4D tensor of shape (N,C,H,W) as input, but it is shape is {input.shape}'
        assert len(weight.shape) == 2, f'shift2d_func(): expected 2D tensor of shape (C,2) as weight, but it is shape is {weight.shape}'
        
        assert input.shape[1] == weight.shape[0],  f'shift2d_func(): expected that input and weight have equal number of channels, but input have {input.shape[1]} and weight have {weight.shape[0]} channels.'
        assert input.device == weight.device, f'shift2d_func(): expected input and weights to be on same device, but input is  on {input.device} and weights is on {weight.device}'

        if input.is_cuda and not SHIFTS_GPU_AVAILABLE:
            raise Exception('shifts on CUDA device is asked, but it seems that it is not available. Please install it')
        if not input.is_cuda and not SHIFTS_CPU_AVAILABLE:
            raise Exception('shifts on CPU is not available. Please install it.')

        ctx.padding_mode = padding_mode
        ctx.active_flag = active_flag
        ctx.save_for_backward(input, weight)
        if input.is_cuda:
            output = shift2d_gpu(input, weight,ctx.padding_mode, ctx.active_flag)
        else:
            output = shift2d_cpu(input, weight,ctx.padding_mode, ctx.active_flag)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        if input.is_cuda:
            grad_input, grad_weight = shift2d_backward_gpu(grad_output, weight, input, ctx.padding_mode, ctx.active_flag)
        else:
            grad_input, grad_weight = shift2d_backward_cpu(grad_output, weight, input, ctx.padding_mode, ctx.active_flag)
        return grad_input, grad_weight, None, None

    
    
    
class Shift1D(nn.Module):
    """
        Performs (index)shift operation under 3D tensor. Zero-FLOPs replacement of Depth-Wise convolution.
        
        Note:Shift values and directions is learnable for each channel.
       
        Arguments:
            in_channels(int) – Number of channels in the input image.
            padding(str) - Padding added to the input during bilinear interpolation.
                           Allowed: ['zeros', 'border', 'reflect', 'symmetric']. Default: 'zeros'.
            init_stride(float) - Border for uniform initialization of weights(shifts): [-init_stride;init_stride]. Default: 1.
            sparsity_term(float) - Strength of sparsity. Default: 5e-4.
            active_shift(bool) - Compute forward pass via bilinear interpolation. Default: False.
    """
    def __init__(self, in_channels, padding='zeros',
                 init_stride = 1,
                 sparsity_term=5e-4,
                 active_flag=False):
        super(Shift1D, self).__init__()
        self.padding_dict = {'zeros':0, 'border':1, 'reflect':2, 'symmetric':3}
        assert padding.lower() in self.padding_dict.keys(), f'incorrect padding option: {padding}'
        self.padding = self.padding_dict[padding]
        self.sparsity_term = sparsity_term
        self.weight = nn.Parameter(torch.Tensor(in_channels))
        self.reset_parameters(init_stride)
        self.__active = active_flag
        
    def reset_parameters(self, init_stride):
        self.weight.data.uniform_(-abs(init_stride), abs(init_stride))
    
    def __compute_weight_loss(self):
        return self.sparsity_term * torch.sum(torch.abs(self.weight))
                
    def forward(self, input):
        if bool(self.sparsity_term):
            return shift1d_func.apply(input, self.weight, self.padding, self.__active), self.__compute_weight_loss()
        return shift1d_func.apply(input, self.weight, self.padding, self.__active)    
    
    
class Shift2D(nn.Module):
    """
        Performs (index)shift operation under 4D tensor(by h and w axes). Zero-FLOPs replacement of Depth-Wise convolution.
        
        Note:Shift values and directions is learnable for each channel.
       
        Arguments:
            in_channels(int) – Number of channels in the input image.
            padding(str) - Padding added to the input during bilinear interpolation.
                           Allowed: ['zeros', 'border', 'reflect', 'symmetric']. Default: 'zeros'.
            init_stride(float) - Border for uniform initialization of weights(shifts): [-init_stride;init_stride]. Default: 1.
            sparsity_term(float) - Strength of sparsity. Default: 5e-4.
            active_shift(bool) - Compute forward pass via bilinear interpolation. Default: False.
    """
    def __init__(self, in_channels, padding='zeros',
                 init_stride = 1,
                 sparsity_term=5e-4,
                 active_flag=False):
        super(Shift2D, self).__init__()
        self.padding_dict = {'zeros':0, 'border':1, 'reflect':2, 'symmetric':3}
        assert padding.lower() in self.padding_dict.keys(), f'incorrect padding option: {padding}'
        self.padding = self.padding_dict[padding]
        self.sparsity_term = sparsity_term
        self.weight = nn.Parameter(torch.Tensor(in_channels,2))
        self.reset_parameters(init_stride)
        self.__active = active_flag
        
    def reset_parameters(self, init_stride):
        self.weight.data.uniform_(-abs(init_stride), abs(init_stride))
    
    def __compute_weight_loss(self):
        return self.sparsity_term * torch.sum(torch.abs(self.weight))
                
    def forward(self, input):
        if bool(self.sparsity_term):
            return shift2d_func.apply(input, self.weight, self.padding, self.__active), self.__compute_weight_loss()
        return shift2d_func.apply(input, self.weight, self.padding ,self.__active)

    
