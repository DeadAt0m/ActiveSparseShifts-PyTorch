import torch
from torch.nn.functional import has_torch_function, handle_torch_function
from pathlib import Path
import platform
from torch.autograd import Function

dispatch_dict = {}

TORCHSHIFTS_CPU_AVAILABLE = True
try:
    module_path = Path(__file__).parent.parent / (f'shifts_cpu.{"pyd" if platform.system() == "Windows" else "so"}')
    torch.ops.load_library(str(module_path))
    shift1d_cpu = torch.ops.shifts_cpu.shift1d_cpu
    shift1d_backward_cpu = torch.ops.shifts_cpu.shift1d_backward_cpu
    shift2d_cpu = torch.ops.shifts_cpu.shift2d_cpu
    shift2d_backward_cpu = torch.ops.shifts_cpu.shift2d_backward_cpu
    shift3d_cpu = torch.ops.shifts_cpu.shift3d_cpu
    shift3d_backward_cpu = torch.ops.shifts_cpu.shift3d_backward_cpu
    quantized_shift1d_cpu = torch.ops.shifts_cpu.q_shift1d_cpu
    quantized_shift2d_cpu = torch.ops.shifts_cpu.q_shift2d_cpu
    quantized_shift3d_cpu = torch.ops.shifts_cpu.q_shift3d_cpu
    dispatch_dict = {
    'shift1d': {'cpu':  {'forward':shift1d_cpu,  'backward':shift1d_backward_cpu},
                'quantized_cpu': quantized_shift1d_cpu},
    'shift2d': {'cpu':  {'forward':shift2d_cpu,  'backward':shift2d_backward_cpu},
                'quantized_cpu': quantized_shift2d_cpu},           
    'shift3d': {'cpu':  {'forward':shift3d_cpu,  'backward':shift3d_backward_cpu},
                'quantized_cpu': quantized_shift3d_cpu},
}
except ImportError:
    TORCHSHIFTS_CPU_AVAILABLE = False

TORCHSHIFTS_CUDA_AVAILABLE = True
try:
    module_path = Path(__file__).parent.parent / (f'shifts_cuda.{"pyd" if platform.system() == "Windows" else "so"}')
    torch.ops.load_library(str(module_path))
    shift1d_cuda = torch.ops.shifts_cuda.shift1d_cuda
    shift1d_backward_cuda = torch.ops.shifts_cuda.shift1d_backward_cuda
    shift2d_cuda = torch.ops.shifts_cuda.shift2d_cuda
    shift2d_backward_cuda = torch.ops.shifts_cuda.shift2d_backward_cuda
    shift3d_cuda = torch.ops.shifts_cuda.shift3d_cuda
    shift3d_backward_cuda = torch.ops.shifts_cpu.shift3d_backward_cuda
    dispatch_dict['shift1d']['cuda'] =  {'forward':shift1d_cuda, 'backward':shift1d_backward_cuda}
    dispatch_dict['shift2d']['cuda'] =  {'forward':shift2d_cuda, 'backward':shift2d_backward_cuda}
    dispatch_dict['shift3d']['cuda'] =  {'forward':shift3d_cuda, 'backward':shift3d_backward_cuda}
except ImportError:
    TORCHSHIFTS_CUDA_AVAILABLE = False

Tensor = torch.Tensor

def shift_dispatcher(dimension, *args):
    assert TORCHSHIFTS_CPU_AVAILABLE, "something went wrong during importing C module"
    is_cuda = args[0].is_cuda
    if is_cuda:
        assert TORCHSHIFTS_CUDA_AVAILABLE, "it seems you has no CUDA device or something went wrong during compiling/importing module, please check your installation"
    tensor_args = list(filter(lambda x: type(x) is Tensor, args))
    _device = 'quantize' if args[0].is_quantized else ('cuda' if is_cuda else 'cpu')
    _type = len(tensor_args) > 2 and _device != 'quantize'
    func = dispatch_dict[f'shift{dimension}d'][_device][_type]
    if not torch.jit.is_scripting():
        if len(tensor_args) == 0 and has_torch_function(args):
            return handle_torch_function(func, tensor_args, *args )
    return func(*args)

def _forward_template(dim, ctx, input, weight, padding_mode, active_flag):
    assert type(padding_mode) == int, f'shift{dim}d_func() expected int padding_mode'
    assert padding_mode in list(range(5)), f'shift{dim}d_func() expected padding_mode can be 0 - zeros, 1 - border, 2 - periodic, 3 - reflect, 4 - symmetric'
    assert len(input.shape) == 2+dim, f'shift{dim}d_func(): expected {2+dim}D tensor as input, but it is shape is {input.shape}'
    assert len(weight.shape) == dim, f'shift{dim}d_func(): expected {dim} tensor as weight, but it is shape is {weight.shape}'
    assert input.shape[1] == weight.shape[0],  f'shift{dim}d_func(): expected that input and weight have equal number of channels, but input have {input.shape[1]} and weight have {weight.shape[0]} channels.'
    assert input.device == weight.device, f'shift{dim}d_func(): expected input and weights to be on same device, but input is  on {input.device} and weights is on {weight.device}'
    ctx.padding_mode = padding_mode
    ctx.active_flag = active_flag
    ctx.save_for_backward(input, weight)
    output = shift_dispatcher(dim, input, weight, padding_mode, active_flag)
    return output

def _backward_template(dim, ctx, grad_output):
    input, weight = ctx.saved_tensors
    grad_input, grad_weight = shift_dispatcher(dim, grad_output, weight, input, ctx.padding_mode, ctx.active_flag)
    return grad_input, grad_weight, None, None 


class shift1d_func(Function):
    @staticmethod
    def forward(ctx, input, weight, padding_mode, active_flag):
        return _forward_template(1, ctx, input, weight, padding_mode, active_flag)
    @staticmethod
    def backward(ctx, grad_output):
        return _backward_template(1, ctx, grad_output)

class shift2d_func(Function):
    @staticmethod
    def forward(ctx, input, weight, padding_mode, active_flag):
        return _forward_template(2, ctx, input, weight, padding_mode, active_flag)
    @staticmethod
    def backward(ctx, grad_output):
        return _backward_template(2, ctx, grad_output)

class shift3d_func(Function):
    @staticmethod
    def forward(ctx, input, weight, padding_mode, active_flag):
        return _forward_template(3, ctx, input, weight, padding_mode, active_flag)
    @staticmethod
    def backward(ctx, grad_output):
        return _backward_template(3, ctx, grad_output)
