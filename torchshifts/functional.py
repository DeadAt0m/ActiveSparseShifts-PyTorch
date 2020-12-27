import torch
from .extension import _assert_has_ops
from typing import Optional

Tensor = torch.Tensor

def shift1d_func(input: Tensor, weights: Tensor,
                 padding_mode: int, active_flag: bool,
                 borders: Optional[Tensor] = None) -> Tensor:
    """
        Performs shift operation on 1D tensor
        Arguments:
            input (Tensor[N, C, H]): input 3D tensor
            weights (Tensor[C, 1]): tensor contained shift(amount(abs) and direction(sign)) value for each channel of 1D tensor
            padding_mode (int): padding applyed during shift. Allowed following modes: 0 - zeros, 
                                                                                       1 - border,
                                                                                       2 - periodic, 
                                                                                       3 - reflective, 
                                                                                       4 - symmetric
            active_flag (bool): if true - the active shift(via billinear interpolation) will used on forward pass.
                                This option has no effect if input is Quantized tensor.
            borders (Tensor[1,2]): dim x (left_border, right_border) output tensor will be cut off proportional to borders 
        Returns:
            output (Tensor[N, C, H])
    """
    _assert_has_ops()
    assert padding_mode in [0,1,2,3,4], f'shift1d_func() expected padding_mode can be 0 - zeros, 1 - border, 2 - periodic, 3 - reflect, 4 - symmetric'
    assert len(input.shape) == 3, f'shift1d_func(): expected 3D tensor as input, but it is shape is {input.shape}'
    assert weights.shape[-1] == 1, f'shift1d_func(): expected [n_channels,1] tensor as weight, but it is shape is {weights.shape}'
    assert input.shape[1] == weights.shape[0],  f'shift1d_func(): expected that input and weight have equal number of channels, but input have {input.shape[1]} and weight have {weights.shape[0]} channels.'
    assert input.device == weights.device, f'shift1d_func(): expected input and weights to be on same device, but input is  on {input.device} and weights is on {weights.device}'
    if borders is not None:
        assert (len(borders.shape) == 2) and (borders.shape[1] == 2) and (borders.shape[0] == 1), f'borders must have shape [1, 2]'
    else:
        borders = torch.Tensor()
    return torch.ops.torchshifts.shift1d(input, weights, borders, padding_mode, active_flag)


def shift2d_func(input: Tensor, weights: Tensor,
                 padding_mode: int, active_flag: bool,
                 borders: Optional[Tensor] = None) -> Tensor:
    """
        Performs shift operation on 2D tensor
        Arguments:
            input (Tensor[N, C, H, W]): input 4D tensor
            weights (Tensor[C, 2]): tensor contained 2 shift(amount(abs) and direction(sign)) values(for H and W axes) for each channel of 2D tensor.
            padding_mode (int): padding applyed during shift. Allowed following modes: 0 - zeros, 
                                                                                       1 - border,
                                                                                       2 - periodic, 
                                                                                       3 - reflective, 
                                                                                       4 - symmetric
            active_flag (bool): if true - the active shift(via billinear interpolation) will used on forward pass.
                                This option has no effect if input is Quantized tensor.
            borders (Tensor[2,2]): dim x (left_border, right_border) output tensor will be cut off proportional to borders 
        Returns:
            output (Tensor[N, C, H. W])
    """
    _assert_has_ops()
    assert padding_mode in [0,1,2,3,4], f'shift2d_func() expected padding_mode can be 0 - zeros, 1 - border, 2 - periodic, 3 - reflect, 4 - symmetric'
    assert len(input.shape) == 4, f'shift2d_func(): expected 4D tensor as input, but it is shape is {input.shape}'
    assert weights.shape[-1] == 2, f'shift2d_func(): expected [n_channels,2] tensor as weight, but it is shape is {weights.shape}'
    assert input.shape[1] == weights.shape[0],  f'shift2d_func(): expected that input and weight have equal number of channels, but input have {input.shape[1]} and weight have {weights.shape[0]} channels.'
    assert input.device == weights.device, f'shift2d_func(): expected input and weights to be on same device, but input is  on {input.device} and weights is on {weights.device}'
    if borders is not None:
        assert (len(borders.shape) == 2) and (borders.shape[1] == 2) and (borders.shape[0] == 2), f'borders must have shape [2, 2]'
    else:
        borders = torch.Tensor()
    return torch.ops.torchshifts.shift2d(input, weights, borders, padding_mode, active_flag)

def shift3d_func(input: Tensor, weights: Tensor,
                 padding_mode: int, active_flag: bool,
                 borders: Optional[Tensor] = None) -> Tensor:
    """
        Performs shift operation on 3D tensor
        Arguments:
            input (Tensor[N, C, H, W, D]): input  5D tensor
            weights (Tensor[C, 3]): tensor contained 3 shift(amount(abs) and direction(sign)) values(for H,W and D axes) for each channel of 3D tensor.
            padding_mode (int): padding applyed during shift. Allowed following modes: 0 - zeros, 
                                                                                       1 - border,
                                                                                       2 - periodic, 
                                                                                       3 - reflective, 
                                                                                       4 - symmetric
            active_flag (bool): if true - the active shift(via billinear interpolation) will used on forward pass.
                                This option has no effect if input is Quantized tensor.
            borders (Tensor[3,2]): dim x (left_border, right_border) output tensor will be cut off proportional to borders 
        Returns:
            output (Tensor[N, C, H, W, D])
    """
    _assert_has_ops()
    assert padding_mode in [0,1,2,3,4], f'shift3d_func() expected padding_mode can be 0 - zeros, 1 - border, 2 - periodic, 3 - reflect, 4 - symmetric'
    assert len(input.shape) == 5, f'shift3d_func(): expected 5D tensor as input, but it is shape is {input.shape}'
    assert weights.shape[-1] == 3, f'shift3d_func(): expected [n_channels,3] tensor as weight, but it is shape is {weights.shape}'
    assert input.shape[1] == weights.shape[0],  f'shift3d_func(): expected that input and weight have equal number of channels, but input have {input.shape[1]} and weight have {weights.shape[0]} channels.'
    assert input.device == weights.device, f'shift3d_func(): expected input and weights to be on same device, but input is  on {input.device} and weights is on {weights.device}'
    if borders is not None:
        assert (len(borders.shape) == 2) and (borders.shape[1] == 2) and (borders.shape[0] == 3), f'borders must have shape [3, 2]'
    else:
        borders = torch.Tensor()
    return torch.ops.torchshifts.shift3d(input, weights, borders, padding_mode, active_flag)
