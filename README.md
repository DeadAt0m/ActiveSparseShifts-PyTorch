PyTorch implementation of Sparse Shift Layer(SSL) for 3D, 4D and 5D tensors  from "All You Need is a Few Shifts: Designing Efficient Convolutional Neural Networks
for Image Classification" (https://arxiv.org/pdf/1903.05285.pdf) 

(**I am not the author** any of mentioned articles, I just implement this for my own purposes)
## !NOW FOR PYTORCH 1.7 ONLY!

## Theory

### [Shift operation](https://en.wikipedia.org/wiki/Shift_operator): 

shifts tensor data(in memory) by indexes. Value and direction of shift are learnable and different between channels.
It might be considered as Zero-FLOP replacement of DepthWise Convolution, wiht 4.5x less memory consumption(in compare wiht 3x3 DepthWise ConvD).

### Articles summary:
* [GroupedShift](https://arxiv.org/pdf/1711.08141.pdf): First known application of shifts operator as replace of depthwise convolution. It utilize shifts as their exact form on forward and backward, hence the shifts values (weights) are not learnable (and for simplicity applied to group of channels, see article for detail) and act like hyperparams.
  
  (Officially we have not support this kind of shifts here, but for exact 

* [Active Shift](https://arxiv.org/pdf/1806.07370.pdf): Replacing shift operation on linear(bi-,tri- for 2D,3D cases) interpolation on both forward and backward pass. "Shifts" values became learnable (because they are floats) and moreover shifts defined for each channel.
  
* [Sparse Shift Layer(SSL)]( (https://arxiv.org/pdf/1903.05285.pdf)): The combination of two above articles. "Shifts" values are still learnable vi interpolation(on backward pass), and use exact shift operator on forward pass ("shift" values just rounded during forward pass). So we have simple Zero-FLOP shift operation (which is also native quantized, because shift operator require integer values), instead of DepthWise convolution! 
    
    Sparse - stands to L1 regularization on weights, this obviously sparsifying the shifts values among channel axis!


    ![alt text](https://github.com/DeadAt0m/ActiveSparseShifts-PyTorch/raw/master/shifts.png "Shifts evolution")

## Implementation details:

* By default all Shift modules are Sparse Shift Layers! The module is always returns  ```output``` and ```loss```, where is last is L1 regularization loss(see theory), which should be added to general loss for take an effect!
  
* Active Shift can be enabled by setting ```active_flag=True```, and ```sparsity_term=0```, because we do not need to compute regularization term(at least in original article).
  
* Grouped Shifts are not officially supported here, however technically it possible: set  ```active_flag=False``` and ```sparsity_term=0```, freeze ```.weights``` params from gradient comptuation like ```shift_layer.weights.requires_grad = False``` (inside C function the gradient for weights will be always computed, so you will not gain in perfomance) and don't forget properly reinit ```.weights``` values(including channels groups, etc.)
  
* We implement several padding variants for filling empty values after shifts:
  Zeros (by default), Border, Periodic(stands for circular shifts!), Reflect and Symmetric. See [here](https://pywavelets.readthedocs.io/en/latest/ref/signal-extension-modes.html) for details.(This paddings is also used during interpolation calculation) 
  

## Requirements:
    C++17 must be supported by your compiler!
    PyTorch >= 1.7.0; 

## Instalation:
1. Clone this repo and ```cd ActiveSparseShifts-PyTorch```
(1b). If you compile with CUDA, please pass path to nvcc to CUDA_HOME env variable!
2. Run ```python setup.py install``` 

    
## Using:

Example:

    from torchshifts import Shift1d, Shift2d, Shift3d
    shift_layer = Shift1d(in_channels=3)

Additional options for shift layer:

    padding(str) - Padding for filling empty values.
                   Allowed: ['zeros', 'border', 'periodic', 'reflect', 'symmetric']. Default: 'zeros'.
    init_shift(float) - Border for uniform initialization of weights(shifts): [-init_shift; init_shift]. Default: 1.
    sparsity_term(float) - Strength of sparsity. Default: 5e-4.
    active_flag(bool) - Enable Active Shift instead of SSL. Default: False
    emulate_dw(dict) - Just pass params of depthwise conv, that you trying replace with shift layer.
                               It applies a heuristic and try to emulate their properties(including output shape)
    init_thumb_rule(int) - Type of thumb rule for shifts initialization. Allowed: Type 1(default): uniform(-init_shift, init_shift),
                                                                                  Type 2: uniform(0,init_shift) * random_sign
                                                                                  
                                                                                

## Additionals:
1. Depthwise Convolution Emulation: 
   Provides a heuristic rules for emulation of DepthWise Convolution via Shift layer
   in terms of output shape and shift kernel behaviour.
        
   a) This directly influence on proper shift param initialization.
   b) Output shape via cutting the output and pooling(depending on stride)
   c) Automaticaly using AveragePooling for emulation stride > 1

2. Pytorch Quantization: SSL shifts can be used in quantized pipeline!
   Shifts do not needed the activation tracking and so model with shift module can be easily converted by following:
    ```
    from torchshifts import quant_mapping
    torch.quantization.convert(<model_with_Shift_module>, ..., mapping=quant_mapping)
    ```
3. Pytorch JIT: We support it out-of-box:
   ``` torch.jit.trace_module(<model_with_Shift_module>) ```


## TO DO:
  1. Add unit tests(yes I still make testing in some strange manners)
  3. Make more weights initialization schemes!
