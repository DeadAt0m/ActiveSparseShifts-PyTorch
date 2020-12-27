from torch import nn
import torch
from torchshifts.functional import shift1d_func, shift2d_func, shift3d_func
import random
from functools import partial

paddings_dict = {'zeros':0, 'border':1, 'periodic':2, 'reflect':3, 'symmetric':4}


def _wrap_dim(val, dim, name):
    if isinstance(val, tuple):
        val = list(val)
    if not isinstance(val, list):
        val = [val] * dim
    if len(val) != dim: 
        print(f'{name} params has different kernel sizes, but length of list do not corresponds to dim: {dim}, and was reduced')
        val = val[:dim]
    return val
    
    
def _create_dw_emulation(args, dim):
    """
        Heuristic rules for emulation of DepthWise Conv via Shift layer
        in terms of output shape and and shift kernel behaviour.
        
        This directly influence on proper shift param initialization.
        Output shape via cutting the output and pooling(depending on stride)
    
    
    """    
    assert isinstance(args,dict), f'args must be dict'
    assert 'kernel_size' in args, f'args must contains at least the kernel_size inside'
    if 'dilation' in args:
        print('Warning! Found the dilation param which is not supported and will be ignored')
    kernel_size = _wrap_dim(args['kernel_size'], dim, 'kernel_size')
    padding = _wrap_dim(args.get('padding', 0), dim, 'padding')
    stride = _wrap_dim(args.get('stride', 1), dim, 'stride')
    itrt_scale = 2 if args['init_thumb_rule_type'] == 1 else 1
    
    borders = None
    padding = torch.tensor(padding, requires_grad=False)
    kernel_size = torch.tensor(kernel_size, requires_grad=False)
    tmp = 2*padding - kernel_size + 1
    if (tmp < 0).any():
        borders = torch.zeros(dim, 2, dtype=torch.long, requires_grad=False)
        borders[tmp<0, 0] = abs(tmp[tmp<0]) // 2
        borders[tmp<0, 1] = abs(tmp[tmp<0]) - borders[tmp<0, 0]

    init_shift = kernel_size // itrt_scale   
    scales = torch.tensor(stride,requires_grad=False).unsqueeze(0)
    
    pad_conv = {'zeros':0, 'replicate': 1, 'circular': 2, 'reflect':3,}
    padding = args.get('padding_mode', -1)
    if isinstance(padding,str):
        padding = pad_conv[padding]
    
    return init_shift, scales, borders, padding


class _Shiftnd(nn.Module):
    """
        Base module for all shifts.
        
        Arguments:
            in_channels(int) – Number of channels in the input image.
            padding(str) - Padding added to the input during shift.
                           Allowed: ['zeros', 'border', 'periodic', 'reflect', 'symmetric']. Default: 'zeros'.
            init_shift(float/Tuple[float]) - Border for uniform initialization of weights(shifts). Default: 1.
            sparsity_term(float) - Strength of sparsity. Default: 5e-4.
            active_shift(bool) - Compute forward pass via bilinear interpolation. Default: False.
            emulate_dw(dict) - Just pass params of depthwise conv, that you trying replace with shift layer.
                               It applies a heuristic and try to emulate their properties(including output shape)
            init_thumb_rule(int) - Type of thumb rule for shifts initialization. Allowed: Type 1(default): uniform(-init_shift, init_shift),
                                                                                          Type 2: uniform(0,init_shift) * random_sign
    """
    @staticmethod
    def _identity(x):
        return x
    
    @staticmethod
    def _pooling(ks, dim):
        if isinstance(ks, torch.Tensor):
            ks = ks.squeeze().cpu().numpy().tolist()
        if dim == 1:
            return partial(torch.nn.functional.avg_pool1d, kernel_size=ks, stride=ks, ceil_mode=True)
        elif dim == 2:
            return partial(torch.nn.functional.avg_pool2d, kernel_size=ks, stride=ks, ceil_mode=True)
        else:
            return partial(torch.nn.functional.avg_pool3d, kernel_size=ks, stride=ks, ceil_mode=True)
    
    @staticmethod
    def _init_thumb_rule_1(size, shape):
        return 2*size*torch.rand(shape) - size

    @staticmethod
    def _init_thumb_rule_2(size, shape):
        return size*torch.rand(shape) * (1 if random.random() < 0.5 else -1)


    def __init__(self, in_channels, padding='zeros',
                 init_shift=1,
                 sparsity_term=5e-4,
                 active_flag=False,
                 emulate_dw=None,
                 init_thumb_rule=1):
        super(_Shiftnd, self).__init__()
        assert padding.lower() in paddings_dict.keys(), f'incorrect padding option: {padding}'
        self.padding = paddings_dict[padding]
        self.sparsity_term = sparsity_term
        self.in_channels = in_channels
        self._active_flag = active_flag
        self._shift_func = self._init_shift_fn()
        self.cut_borders = None
        self._reduction_fn = self._identity
        # init weights
        self._w_init_func = self._init_thumb_rule_1
        if init_thumb_rule == 2:
            self._w_init_func == self._init_thumb_rule_2
        # init hyper params
        self.init_shift = torch.tensor(_wrap_dim(init_shift, self.dim, 'init_shift'), 
                                       requires_grad=False)
        self._w_post_init_scale = torch.ones(1, self.dim, requires_grad=False)
        
        if emulate_dw is not None:
            emulate_dw['init_thumb_rule_type'] = init_thumb_rule
            out = _create_dw_emulation(emulate_dw, self.dim)
            self.init_shift, self._w_post_init_scale, self.cut_borders, padding = out
            if padding != -1:
                self.padding == padding
            if not (self._w_post_init_scale == 1).all():
                self._reduction_fn = self._pooling(self._w_post_init_scale, self.dim)
        self._init_weights()


    def _init_shift_fn(self):
        raise NotImplemented

    def _init_weights(self):
        self.weight = nn.Parameter(torch.Tensor(self.in_channels, self.dim))
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.dim):
            self.weight.data[:,i] = self._w_init_func(self.init_shift[i], self.in_channels)
        self.weight.data *= self._w_post_init_scale

    def _compute_weight_loss(self):
        return self.sparsity_term * torch.sum(torch.abs(self.weight))

    def forward(self, input):
        loss = self._compute_weight_loss() if bool(self.sparsity_term) else None
        out = self._shift_func(input, self.weight, self.padding, self._active_flag, self.cut_borders)
        return self._reduction_fn(out), loss

    def extra_repr(self):
        pad = dict(zip(paddings_dict.values(), paddings_dict.keys()))[self.padding]
        active = f'Active shift on forward pass: {"Yes" if self._active_flag else "No"}'
        sp = f'Sparse shift: {"Yes - sparsity strength: {}".format(self.sparsity_term) if bool(self.sparsity_term) else "No"}'
        return f'in_channels={self.in_channels}, padding_method={pad}, {active}, {sp}'


    
class Shift1d(_Shiftnd):
    """
        Performs (index)shift operation under 3D tensor. Zero-FLOPs replacement of Depth-Wise convolution.
        

        Notes: 
            - Shift values and directions is learnable for each channel.
            - Forward method is always return the two terms: output and loss
            - loss is None if sparsity_term  is greater than zero


        Arguments:
            in_channels(int) – Number of channels in the input image.
            padding(str) - Padding added to the input during shift.
                           Allowed: ['zeros', 'border', 'periodic', 'reflect', 'symmetric']. Default: 'zeros'.
            init_shift(float) - Border for uniform initialization of weights(shifts). Default: 1.
            sparsity_term(float) - Strength of sparsity. Default: 5e-4.
            active_shift(bool) - Compute forward pass via bilinear interpolation. Default: False.
            emulate_dw(dict) - Just pass params of depthwise conv, that you trying replace with shift layer.
                               It applies a heuristic and try to emulate their properties(including output shape)
            init_thumb_rule(int) - Type of thumb rule for shifts initialization. Allowed: Type 1(default): uniform(-init_shift, init_shift),
                                                                                          Type 2: uniform(0,init_shift) * random_sign
    """
    def __init__(self, in_channels, padding='zeros',
                 init_shift = 1, sparsity_term=5e-4, active_flag=False,
                 emulate_dw=None,
                 init_thumb_rule=1):
        self.dim = 1
        super(Shift1d, self).__init__(in_channels, padding, init_shift, sparsity_term,
                                      active_flag, emulate_dw, init_thumb_rule)
    
    def _init_shift_fn(self):
        return shift1d_func
    
class Shift2d(_Shiftnd):
    """
        Performs (index)shift operation under 4D(by h and w axes) tensor. Zero-FLOPs replacement of Depth-Wise convolution.
        

        Notes: 
            - Shift values and directions is learnable for each channel.
            - Forward method is always return the two terms: output and loss
            - loss is None if sparsity_term  is greater than zero


        Arguments:
            in_channels(int) – Number of channels in the input image.
            padding(str) - Padding added to the input during shift.
                           Allowed: ['zeros', 'border', 'periodic', 'reflect', 'symmetric']. Default: 'zeros'.
            init_stride(float) - Border for uniform initialization of weights(shifts). Default: 1.
            sparsity_term(float) - Strength of sparsity. Default: 5e-4.
            active_shift(bool) - Compute forward pass via bilinear interpolation. Default: False.
            emulate_dw(dict) - Just pass params of depthwise conv, that you trying replace with shift layer.
                               It applies a heuristic and try to emulate their properties(including output shape)
            init_thumb_rule(int) - Type of thumb rule for shifts initialization. Allowed: Type 1(default): uniform(-init_shift, init_shift),
                                                                                          Type 2: uniform(0,init_shift) * random_sign
    """
    def __init__(self, in_channels, padding='zeros',
                 init_shift = 1, sparsity_term=5e-4, active_flag=False,
                 emulate_dw=None,
                 init_thumb_rule=1):
        self.dim = 2
        super(Shift2d, self).__init__(in_channels, padding, init_shift, sparsity_term,
                                      active_flag, emulate_dw, init_thumb_rule)
    
    def _init_shift_fn(self):
        return shift2d_func
    

class Shift3d(_Shiftnd):
    """
        Performs (index)shift operation under 5D(by h, w and d axes) tensor. Zero-FLOPs replacement of Depth-Wise convolution.
        

        Notes: 
            - Shift values and directions is learnable for each channel.
            - Forward method is always return the two terms: output and loss
            - loss is None if sparsity_term  is greater than zero


        Arguments:
            in_channels(int) – Number of channels in the input image.
            padding(str) - Padding added to the input during shift.
                           Allowed: ['zeros', 'border', 'periodic', 'reflect', 'symmetric']. Default: 'zeros'.
            init_stride(float) - Border for uniform initialization of weights(shifts). Default: 1.
            sparsity_term(float) - Strength of sparsity. Default: 5e-4.
            active_shift(bool) - Compute forward pass via bilinear interpolation. Default: False.
            emulate_dw(dict) - Just pass params of depthwise conv, that you trying replace with shift layer.
                               It applies a heuristic and try to emulate their properties(including output shape)
            init_thumb_rule(int) - Type of thumb rule for shifts initialization. Allowed: Type 1(default): uniform(-init_shift, init_shift),
                                                                                          Type 2: uniform(0,init_shift) * random_sign
    """
    def __init__(self, in_channels, padding='zeros',
                 init_shift = 1, sparsity_term=5e-4, active_flag=False,
                 emulate_dw=None,
                 init_thumb_rule=1):
        self.dim = 3
        super(Shift3d, self).__init__(in_channels, padding, init_shift, sparsity_term,
                                      active_flag, emulate_dw, init_thumb_rule)
    
    def _init_shift_fn(self):
        return shift3d_func
    
