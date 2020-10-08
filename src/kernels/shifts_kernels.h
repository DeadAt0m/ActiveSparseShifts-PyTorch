#include "interpolation.h"
enum class BIPadding {Zeros, Border, Periodic, Reflect, Symmetric};

#define STATIC_IF(e) if constexpr(e)
#define STATIC_ELSEIF(e) else if constexpr(e)
#define STATIC_ELSE else
#define STATIC_ENDIF 

template<typename T>
FTYPE T mod(T a, T b){return (b + (a % b)) % b;}

template<typename idx_t>
FTYPE idx_t infer_index(idx_t index, idx_t len, BIPadding padding_mode){
    if (len == 1){return 0;}
    //     (len == 1) is used just for bypassing, in case when tensor dimension are not available 
    if ((index < len) && (index >= 0)) {return index;};
    idx_t out_index = index;
    bool odd_seq;
    switch (padding_mode){        
        case BIPadding::Zeros: 
            out_index = -1;
            break;
        case BIPadding::Border:
            out_index = (out_index >= len) ? (len - 1) : 0;
            break;
        case BIPadding::Periodic:
            out_index = mod<idx_t>(out_index, len);
            break;
        case BIPadding::Reflect:
            odd_seq = (out_index / (len - 1)) & 1;
            out_index = mod<idx_t>(out_index, len - 1);
            if (odd_seq){out_index = len - 1 - out_index;}
            break;
        case BIPadding::Symmetric:
            odd_seq = (out_index / len) & 1;
            out_index = mod<idx_t>(out_index, len);
            if (odd_seq){out_index = len - 1 - out_index;}
            break;
    }
    return out_index;
}

template<typename scalar_t, typename idx_t>
FTYPE void get_shifted_value(idx_t i_shifted, idx_t sizeH, idx_t strideH,
                                 idx_t j_shifted, idx_t sizeW, idx_t strideW,
                                 idx_t k_shifted, idx_t sizeD, idx_t strideD,
                                 idx_t c, idx_t strideC,
                                 scalar_t* array, scalar_t zero_point, 
                                 BIPadding padding_mode, scalar_t output_value){
    output_value = zero_point;
    idx_t tidx_i = 0;
    idx_t tidx_j = 0;
    idx_t tidx_k = 0;
    tidx_i = infer_index<idx_t>(i_shifted, sizeH, padding_mode);
    tidx_j = infer_index<idx_t>(j_shifted, sizeW, padding_mode);
    tidx_k = infer_index<idx_t>(k_shifted, sizeD, padding_mode); 
    if ((tidx_i>=0)&&(tidx_j>=0)&&(tidx_k>=0)){ output_value = array[tidx_i*strideH+tidx_j*strideW+tidx_k*strideD+c*strideC];}
}

template<typename scalar_t, typename idx_t>
FTYPE void get_shifted_values(idx_t i_shifted, idx_t sizeH, idx_t strideH,
                              idx_t j_shifted, idx_t sizeW, idx_t strideW,
                              idx_t k_shifted, idx_t sizeD, idx_t strideD,
                              idx_t c, idx_t strideC,
                              scalar_t* array, scalar_t zero_point, 
                              BIPadding padding_mode, scalar_t* output_values){
    get_shifted_value(i_shifted, sizeH, strideH, j_shifted, sizeW, strideW,
                      k_shifted, sizeD, strideD, c, strideC, array, zero_point, padding_mode, output_values[0]);
    get_shifted_value(i_shifted+1, sizeH, strideH, j_shifted, sizeW, strideW,
                      k_shifted, sizeD, strideD, c, strideC, array, zero_point, padding_mode, output_values[1]);
    if (sizeW>1){
        get_shifted_value(i_shifted, sizeH, strideH, j_shifted+1, sizeW, strideW,
                          k_shifted, sizeD, strideD, c, strideC, array, zero_point, padding_mode, output_values[2]);
        get_shifted_value(i_shifted+1, sizeH, strideH, j_shifted+1, sizeW, strideW,
                          k_shifted, sizeD, strideD, c, strideC, array, zero_point, padding_mode, output_values[3]);                           
    } else {
        output_values[2] = zero_point;
        output_values[3] = zero_point;
    }
    if (sizeD>1){
        get_shifted_value(i_shifted, sizeH, strideH, j_shifted, sizeW, strideW,
                          k_shifted+1, sizeD, strideD, c, strideC, array, zero_point, padding_mode, output_values[4]);
        get_shifted_value(i_shifted+1, sizeH, strideH, j_shifted, sizeW, strideW,
                          k_shifted+1, sizeD, strideD, c, strideC, array, zero_point, padding_mode, output_values[5]);
        get_shifted_value(i_shifted, sizeH, strideH, j_shifted+1, sizeW, strideW,
                          k_shifted+1, sizeD, strideD, c, strideC, array, zero_point, padding_mode, output_values[6]);
        get_shifted_value(i_shifted+1, sizeH, strideH, j_shifted+1, sizeW, strideW,
                          k_shifted+1, sizeD, strideD, c, strideC, array, zero_point, padding_mode, output_values[7]);
    } else {
        output_values[4] = zero_point;
        output_values[5] = zero_point;
        output_values[6] = zero_point;
        output_values[7] = zero_point;  
    }                                 
}

template <typename scalar_t, typename idx_t, bool reverse=false>
FTYPE void compute_interpolated(scalar_t* v, scalar_t diff_shiftH, scalar_t diff_shiftW, scalar_t diff_shiftD,
                                idx_t sizeH, idx_t sizeW, idx_t sizeD, scalar_t zero_point, scalar_t output_val){
    scalar_t rcoeff = static_cast<scalar_t>((reverse)?-1:1);
    output_val = zero_point;
    if (sizeD>1){output_val=interp3D(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7],
                                     rcoeff*diff_shiftH, rcoeff*diff_shiftW, rcoeff*diff_shiftD);}
    else if (sizeW>1){output_val=interp2D(v[0], v[1], v[2], v[3], 
                                          rcoeff*diff_shiftH, rcoeff*diff_shiftW);}
    else {output_val=interp1D(v[0], v[1], rcoeff*diff_shiftH);}
}

template <typename scalar_t, typename idx_t>
FTYPE void compute_weight_gradients(scalar_t* v, scalar_t diff_shiftH, scalar_t diff_shiftW, scalar_t diff_shiftD,
                                    idx_t sizeH, idx_t sizeW, idx_t sizeD, scalar_t zero_point, scalar_t* output_grad){
    if (sizeD>1){
        output_grad[0]=interp3D_dx(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7],
                                   diff_shiftW, diff_shiftD);
        output_grad[1]=interp3D_dy(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7],
                                   diff_shiftH, diff_shiftD);
        output_grad[2]=interp3D_dz(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7],
                                   diff_shiftH, diff_shiftW);
    }
    else if (sizeW>1){
        output_grad[0]=interp2D_dx(v[0], v[1], v[2], v[3], 
                                   diff_shiftW);
        output_grad[1]=interp2D_dy(v[0], v[1], v[2], v[3], 
                                   diff_shiftH);
        output_grad[2]=zero_point;
        
    }
    else if (sizeH>1){
        output_grad[0]=interp1D_dx(v[0], v[1]);
        output_grad[1]=zero_point;
        output_grad[2]=zero_point;
    }
}

template <typename scalar_t, typename idx_t, bool quantized, bool active>
FTYPE void shift_forward_kernel_nchwd(scalar_t* input, scalar_t* output,
                                      idx_t* weights, scalar_t* dweights,
                                      idx_t n, idx_t c, idx_t i, idx_t j, idx_t k,
                                      idx_t sizeH, idx_t sizeW, idx_t sizeD,
                                      idx_t input_sN, idx_t input_sC, idx_t input_sH, idx_t input_sW, idx_t input_sD,
                                      idx_t output_sN, idx_t output_sC, idx_t output_sH, idx_t output_sW, idx_t output_sD,
                                      idx_t weights_sC, idx_t weights_sS, idx_t dweights_sC, idx_t dweights_sS,
                                      scalar_t zero_point, idx_t weights_zero_point, BIPadding padding_mode){
    scalar_t *input_NC = input + n*input_sN + c*input_sC;
    scalar_t *output_NCHWD= output + n*output_sN + c*output_sC + i*output_sH + j*output_sW + k*output_sD;
    scalar_t val;
    idx_t shifts[3] = {*(weights+c*weights_sC) - weights_zero_point, 0, 0};
    if (sizeW>1){shifts[1] = *(weights+c*weights_sC+weights_sS) - weights_zero_point;}
    if (sizeD>1){shifts[2] = *(weights+c*weights_sC+2*weights_sC) - weights_zero_point;}
    STATIC_IF((quantized)||(!active))
    {
        get_shifted_value<scalar_t,idx_t>(i+shifts[0], sizeH, input_sH,
                                          j+shifts[1], sizeW, input_sW,
                                          k+shifts[2], sizeD, input_sD,
                                          0, 0, input_NC, zero_point, padding_mode, val);
    } STATIC_ELSE
    {
        scalar_t _vals_array[8];
        get_shifted_values<scalar_t,idx_t>(i+shifts[0], sizeH, input_sH,
                                           j+shifts[1], sizeW, input_sW,
                                           k+shifts[2], sizeD, input_sD,
                                           0, 0, input_NC, zero_point, padding_mode, _vals_array);
        scalar_t dshifts[3] = {*(dweights + c*dweights_sC), 0, 0};
        if (sizeW>1){dshifts[1] = *(dweights + c*dweights_sC + dweights_sS);}
        if (sizeD>1){dshifts[2] = *(dweights + c*dweights_sC + 2*dweights_sS);}
        compute_interpolated<scalar_t,idx_t,false>(_vals_array, dshifts[0], dshifts[1], dshifts[2],
                                                   sizeH, sizeW, sizeD, zero_point, val);
    } STATIC_ENDIF
    *output_NCHWD = val;
}

template <typename scalar_t, typename idx_t, bool active>
FTYPE void shift_backward_kernel_nchwd(scalar_t* input_grad, scalar_t* input,  scalar_t* output_grad,
                                       idx_t* weights, scalar_t* dweights, scalar_t* weights_grad,
                                       idx_t n, idx_t c, idx_t i, idx_t j, idx_t k,
                                       idx_t sizeH, idx_t sizeW, idx_t sizeD,
                                       idx_t input_grad_sN, idx_t input_grad_sC, idx_t input_grad_sH, idx_t input_grad_sW, idx_t input_grad_sD,
                                       idx_t input_sN, idx_t input_sC, idx_t input_sH, idx_t input_sW, idx_t input_sD,
                                       idx_t output_grad_sN, idx_t output_grad_sC, idx_t output_grad_sH, idx_t output_grad_sW, idx_t output_grad_sD,
                                       idx_t weights_sC, idx_t weights_sS, idx_t dweights_sC, idx_t dweights_sS, idx_t weights_grad_sC, idx_t weights_grad_sS,
                                       BIPadding padding_mode){
    scalar_t *input_grad_NC = input_grad + n*input_grad_sN + c*input_grad_sC;
    scalar_t input_grad_NCHWD_val = input_grad_NC[i*input_grad_sH + j*input_grad_sW + k*input_grad_sD];
    scalar_t *input_NC = input + n*input_sN + c*input_sC;
    scalar_t *output_grad_NCHWD= output_grad + n*output_grad_sN + c*output_grad_sC + i*output_grad_sH + j*output_grad_sW + k*output_grad_sD;                                   
    scalar_t zp = static_cast<scalar_t>(0);
    scalar_t _vals_array[8];
    idx_t shifts[3] = {*(weights + c*weights_sC), 0, 0};
    scalar_t dshifts[3] = {*(dweights + c*dweights_sC), zp, zp};
    if (sizeW>1){
        shifts[1] = *(weights + c*weights_sC + weights_sS);       
        dshifts[1] = *(dweights + c*dweights_sC + dweights_sS);}
    if (sizeD>1){
        shifts[2] = *(weights + c*weights_sC + 2*weights_sS);        
        dshifts[2] = *(dweights + c*dweights_sC + 2*dweights_sS);}
    STATIC_IF(active)
    {
        get_shifted_values<scalar_t,idx_t>(i-shifts[0], sizeH, input_sH,
                                           j-shifts[1], sizeW, input_sW,
                                           k-shifts[2], sizeD, input_sD,
                                           0, 0, input_grad_NC, zp, padding_mode, _vals_array);
        compute_interpolated<scalar_t,idx_t, true>(_vals_array, dshifts[0], dshifts[1], dshifts[2],
                                                   sizeH, sizeW, sizeD, zp, *output_grad_NCHWD);
    } STATIC_ELSE
    {                                                              
        get_shifted_value<scalar_t,idx_t>(i-shifts[0], sizeH, input_sH,
                                          j-shifts[1], sizeW, input_sW,
                                          k-shifts[2], sizeD, input_sD,
                                          0, 0, input_grad_NC, zp, padding_mode, *output_grad_NCHWD);
    } STATIC_ENDIF
    get_shifted_values<scalar_t,idx_t>(i+shifts[0], sizeH, input_sH,
                                       j+shifts[1], sizeW, input_sW,
                                       k+shifts[2], sizeD, input_sD,
                                       0, 0, input_NC, zp, padding_mode, _vals_array);
    scalar_t _new_weights_grad[3];                                   
    compute_weight_gradients<scalar_t,idx_t>(_vals_array, dshifts[0], dshifts[1], dshifts[2],
                                             sizeH, sizeW, sizeD, zp, _new_weights_grad);
    ADD((weights_grad + c*weights_grad_sC),(input_grad_NCHWD_val * _new_weights_grad[0]));
    if (sizeW>1){ADD((weights_grad + c*weights_grad_sC + weights_grad_sS),(input_grad_NCHWD_val * _new_weights_grad[1]));}
    if (sizeD>1){ADD((weights_grad + c*weights_grad_sC + 2*weights_grad_sS),(input_grad_NCHWD_val * _new_weights_grad[2]));}
}


template <typename scalar_t, typename idx_t, bool quantized, bool active>
FTYPE void shift_forward_kernel_nhwdc(scalar_t* input, scalar_t* output, 
                                      idx_t* weights, scalar_t* dweights,
                                      idx_t n, idx_t i, idx_t j, idx_t k,
                                      idx_t sizeC, idx_t sizeH, idx_t sizeW, idx_t sizeD,
                                      idx_t input_sN, idx_t input_sC, idx_t input_sH, idx_t input_sW, idx_t input_sD,
                                      idx_t output_sN, idx_t output_sC, idx_t output_sH, idx_t output_sW, idx_t output_sD,
                                      idx_t weights_sC, idx_t weights_sS, idx_t dweights_sC, idx_t dweights_sS,
                                      scalar_t zero_point, idx_t weights_zero_point, BIPadding padding_mode){
    scalar_t *input_N = input + n*input_sN;
    scalar_t *output_NHWD = output + n*output_sN + i*output_sH + j*output_sW + k*output_sD;
    scalar_t val;
    idx_t shifts[3] = {0, 0, 0};
    for (idx_t c = 0; c < sizeC; c++)
    {
        shifts[0] = *(weights+c*weights_sC) - weights_zero_point;
        if (sizeW>1){shifts[1] = *(weights+weights_sS+c*weights_sC) - weights_zero_point;}
        if (sizeD>1){shifts[2] = *(weights+2*weights_sS+c*weights_sC) - weights_zero_point;}
        STATIC_IF((quantized)||(!active))
        {
            get_shifted_value<scalar_t,idx_t>(i+shifts[0], sizeH, input_sH,
                                              j+shifts[1], sizeW, input_sW,
                                              k+shifts[2], sizeD, input_sD,
                                              c, input_sC, input_N, zero_point, padding_mode, val);
        } STATIC_ELSE
        {   
            scalar_t _vals_array[8];
            get_shifted_values<scalar_t,idx_t>(i+shifts[0], sizeH, input_sH,
                                               j+shifts[1], sizeW, input_sW,
                                               k+shifts[2], sizeD, input_sD,
                                               c, input_sC, input_N, zero_point, padding_mode, _vals_array);

            compute_interpolated<scalar_t,idx_t,false>(_vals_array, *(dweights+c*dweights_sC), *(dweights+dweights_sS+c*dweights_sC),
                                                       *(dweights+2*dweights_sS+c*dweights_sC),
                                                       sizeH, sizeW, sizeD, zero_point, val);
        } STATIC_ENDIF
        output_NHWD[c*output_sC] = val;
    }
}

template <typename scalar_t, typename idx_t, bool active>
FTYPE void shift_backward_kernel_nhwdc(scalar_t* input_grad, scalar_t* input,  scalar_t* output_grad,
                                       idx_t* weights, scalar_t* dweights, scalar_t* weights_grad,
                                       idx_t n, idx_t i, idx_t j, idx_t k,
                                       idx_t sizeC, idx_t sizeH, idx_t sizeW, idx_t sizeD,
                                       idx_t input_grad_sN, idx_t input_grad_sC, idx_t input_grad_sH, idx_t input_grad_sW, idx_t input_grad_sD,
                                       idx_t input_sN, idx_t input_sC, idx_t input_sH, idx_t input_sW, idx_t input_sD,
                                       idx_t output_grad_sN, idx_t output_grad_sC, idx_t output_grad_sH, idx_t output_grad_sW, idx_t output_grad_sD,
                                       idx_t weights_sC, idx_t weights_sS, idx_t dweights_sC, idx_t dweights_sS, idx_t weights_grad_sC, idx_t weights_grad_sS,
                                       BIPadding padding_mode){
    scalar_t *input_grad_N = input_grad + n*input_grad_sN;
    scalar_t *input_N = input + n*input_sN;
    scalar_t *output_grad_NHWD= output_grad + n*output_grad_sN + i*output_grad_sH + j*output_grad_sW + k*output_grad_sD;         
    scalar_t *input_grad_NHWD = input_grad_N + i*input_grad_sH + j*input_grad_sW + k*input_grad_sD;
    scalar_t input_grad_NHWDC_val; 
    scalar_t zp = static_cast<scalar_t>(0);
    idx_t shifts[3] = {0, 0, 0};
    scalar_t dshifts[3] = {zp, zp, zp};
    scalar_t _vals_array[8];
    scalar_t _new_weights_grad[3];
    for (idx_t c = 0; c < sizeC; c++)
    {
        shifts[0] = *(weights + c*weights_sC);
        dshifts[3] = *(dweights + c*dweights_sC);
        if (sizeW>1){
            shifts[1] = *(weights+weights_sS+c*weights_sC);         
            dshifts[1] = *(dweights+dweights_sS+c*dweights_sC);}
        if (sizeD>1){
            shifts[2] =  *(weights+2*weights_sS+c*weights_sC);         
            dshifts[2] = *(dweights+2*dweights_sS+c*dweights_sC);}
        STATIC_IF(active)
        {
            get_shifted_values<scalar_t,idx_t>(i-shifts[0], sizeH, input_sH,
                                               j-shifts[1], sizeW, input_sW,
                                               k-shifts[2], sizeD, input_sD,
                                               c, input_grad_sC, input_grad_N, zp, padding_mode, _vals_array);
            compute_interpolated<scalar_t,idx_t,true>(_vals_array, dshifts[0], dshifts[1], dshifts[2],
                                                      sizeH, sizeW, sizeD, zp, output_grad_NHWD[c*output_grad_sC]);
        } STATIC_ELSE
        {
            get_shifted_value<scalar_t,idx_t>(i-shifts[0], sizeH, input_sH,
                                              j-shifts[1], sizeW, input_sW,
                                              k-shifts[2], sizeD, input_sD,
                                              c, input_grad_sC, input_grad_N, zp, padding_mode, output_grad_NHWD[c*output_grad_sC]);
        } STATIC_ENDIF
        get_shifted_values<scalar_t,idx_t>(i+shifts[0], sizeH, input_sH,
                                           j+shifts[1], sizeW, input_sW,
                                           k+shifts[2], sizeD, input_sD,
                                           c, input_sC, input_N, zp, padding_mode, _vals_array);
        compute_weight_gradients<scalar_t,idx_t>(_vals_array, dshifts[0], dshifts[1], dshifts[2],
                                                 sizeH, sizeW, sizeD, zp, _new_weights_grad);
        input_grad_NHWDC_val = input_grad_NHWD[c*input_grad_sC];
        ADD((weights_grad + c*weights_grad_sC),(input_grad_NHWDC_val * _new_weights_grad[0]));
        if (sizeW>1){ADD((weights_grad + weights_grad_sS + c*weights_grad_sC),(input_grad_NHWDC_val * _new_weights_grad[1]));}
        if (sizeD>1){ADD((weights_grad + 2*weights_grad_sS + c*weights_grad_sC),(input_grad_NHWDC_val * _new_weights_grad[2]));}
    }
}


template <typename in_t, typename out_t, bool q_type, bool active>
FTYPE void init_weights(in_t* src, out_t* dst, const int size)
{
    dst = new out_t[size];
    for (int i = 0; i < size; ++i) 
    {
        STATIC_IF(q_type){
            dst[i] = static_cast<out_t>(src[i].val_);
        } STATIC_ELSEIF(active){
            dst[i] = static_cast<out_t>(FLOOR(src[i]));
        } STATIC_ELSE {
            dst[i] = static_cast<out_t>(ROUND(src[i]));
        } STATIC_ENDIF 
    }
} 



template <typename scalar_t>
FTYPE void init_weight_offsets(scalar_t* src, scalar_t* dst, const int size)
{
    dst = new scalar_t[size];
    for (int i = 0; i < size; ++i) 
    {
        dst[i] = src[i] - FLOOR(src[i]);
    }
}