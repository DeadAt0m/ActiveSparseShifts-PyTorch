from torchshifts.functional import shift_dispatcher

def shift1d_quantized(input, weight, padding_mode):
    if not input.is_quantized:
        raise ValueError("Input to 'shift1d_quantized' must be quantized!")
    return shift_dispatcher(1, input, weight, padding_mode)

def shift2d_quantized(input, weight, padding_mode):
    if not input.is_quantized:
        raise ValueError("Input to 'shift2d_quantized' must be quantized!")
    return shift_dispatcher(2, input, weight, padding_mode)

def shift3d_quantized(input, weight, padding_mode):
    if not input.is_quantized:
        raise ValueError("Input to 'shift3d_quantized' must be quantized!")
    return shift_dispatcher(3, input, weight, padding_mode)