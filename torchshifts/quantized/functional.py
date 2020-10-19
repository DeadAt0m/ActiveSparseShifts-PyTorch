from torchshifts.functional import shift1d_func, shift2d_func, shift3d_func

def shift1d_quantized(input, weight, padding_mode):
    if not input.is_quantized:
        raise ValueError("Input to 'shift1d_quantized' must be quantized!")
    return shift1d_func(input, weight, padding_mode, False)

def shift2d_quantized(input, weight, padding_mode):
    if not input.is_quantized:
        raise ValueError("Input to 'shift2d_quantized' must be quantized!")
    return shift2d_func(input, weight, padding_mode, False)

def shift3d_quantized(input, weight, padding_mode):
    if not input.is_quantized:
        raise ValueError("Input to 'shift3d_quantized' must be quantized!")
    return shift3d_func(input, weight, padding_mode, False)