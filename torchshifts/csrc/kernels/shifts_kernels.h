#include "../global_scope.h"
#include "interpolation.h"


enum class BIPadding {Zeros, Border, Periodic, Reflect, Symmetric};

template<typename T>
API_INLINE T mod(T a, T b){return (b + (a % b)) % b;}

template<typename idx_t, BIPadding padding_mode = BIPadding::Zeros>
API_INLINE idx_t infer_index(idx_t index, idx_t len){
    if ((index < len) && (index >= 0)) {return index;};
    idx_t out_index = index;
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
            if (len == 1) {return 0;}
            bool odd_seq = ((idx_t)(out_index<0) + (ABS(out_index)-(idx_t)(out_index<0))/ (len-1)) & 1;
            out_index = mod<idx_t>(out_index, len - 1);
            if (odd_seq){out_index = len - 1 - out_index;}
            break;
        case BIPadding::Symmetric:
            bool odd_seq = ((idx_t)(out_index<0) + (ABS(out_index)-(idx_t)(out_index<0))/ len) & 1;
            out_index = mod<idx_t>(out_index, len);
            if (odd_seq){out_index = len - 1 - out_index;}
            break;
    }
    return out_index;
}

template<typename scalar_t, typename idx_t, 
         int kSpatialDim = 1,
         BIPadding padding_mode = BIPadding::Zeros>
API_INLINE scalar_t get_shifted_value(idx_t i_shifted, idx_t sizeH, idx_t strideH,
                                      idx_t j_shifted, idx_t sizeW, idx_t strideW,
                                      idx_t k_shifted, idx_t sizeD, idx_t strideD,
                                      idx_t c, idx_t strideC,
                                      idx_t i_left_border, idx_t j_left_border, idx_t k_left_border,
                                      idx_t i_right_border, idx_t j_right_border, idx_t k_right_border,
                                      scalar_t* array, scalar_t zero_point){
    scalar_t output_value = zero_point;
    idx_t tidx_i = infer_index<idx_t,padding_mode>(i_shifted + i_left_border, MAX(sizeH, i_right_border));
    idx_t tidx_j = 0;
    idx_t tidx_k = 0;
    bool pass_cond = (tidx_i>=0)&&(tidx_i >= i_left_border)&&(tidx_i < i_right_border);
    if (kSpatialDim > 1){
        tidx_j = infer_index<idx_t,padding_mode>(j_shifted + j_left_border, MAX(sizeW, j_right_border));
        pass_cond *= (tidx_j>=0)&&(tidx_j >= j_left_border)&&(tidx_j < j_right_border);
    }
    if (kSpatialDim > 2){
        tidx_k = infer_index<idx_t,padding_mode>(k_shifted + k_left_border, MAX(sizeD, k_right_border));
        econd_k *= (tidx_k>=0)&&(tidx_k >= k_left_border)&&(tidx_k < k_right_border);
    }
    if (pass_cond){
        output_value = array[tidx_i * strideH + tidx_j * strideW + tidx_k * strideD + c * strideC];
    }
    return output_value;
}

template<typename scalar_t, typename idx_t,
         int kSpatialDim = 1,
         BIPadding padding_mode = BIPadding::Zeros>
API_INLINE void get_shifted_values(idx_t i_shifted, idx_t sizeH, idx_t strideH,
                                   idx_t j_shifted, idx_t sizeW, idx_t strideW,
                                   idx_t k_shifted, idx_t sizeD, idx_t strideD,
                                   idx_t c, idx_t strideC,
                                   idx_t i_left_border, idx_t j_left_border, idx_t k_left_border,
                                   idx_t i_right_border, idx_t j_right_border, idx_t k_right_border,
                                   scalar_t* array, scalar_t zero_point, scalar_t* output_values){
    output_values[0] = get_shifted_value<scalar_t, idx_t, kSpatialDim, padding_mode>(
                                         i_shifted, sizeH, strideH, j_shifted, sizeW, strideW,
                                         k_shifted, sizeD, strideD, c, strideC,
                                         i_left_border, j_left_border, k_left_border,
                                         i_right_border, j_right_border, k_right_border,
                                         array, zero_point);
    output_values[1] = get_shifted_value<scalar_t, idx_t, kSpatialDim, padding_mode>(
                                         i_shifted+1, sizeH, strideH, j_shifted, sizeW, strideW,
                                         k_shifted, sizeD, strideD, c, strideC, 
                                         i_left_border, j_left_border, k_left_border,
                                         i_right_border, j_right_border, k_right_border,
                                         array, zero_point);
    if (kSpatialDim > 1){
        output_values[2] = get_shifted_value<scalar_t, idx_t, kSpatialDim, padding_mode>(
                                         i_shifted, sizeH, strideH, j_shifted+1, sizeW, strideW,
                                         k_shifted, sizeD, strideD, c, strideC,
                                         i_left_border, j_left_border, k_left_border,
                                         i_right_border, j_right_border, k_right_border,
                                         array, zero_point);
        output_values[3] = get_shifted_value<scalar_t, idx_t, kSpatialDim, padding_mode>(
                                         i_shifted+1, sizeH, strideH, j_shifted+1, sizeW, strideW,
                                         k_shifted, sizeD, strideD, c, strideC,
                                         i_left_border, j_left_border, k_left_border,
                                         i_right_border, j_right_border, k_right_border,
                                         array, zero_point);                           
    }
    if (kSpatialDim > 2){
        output_values[4] = get_shifted_value<scalar_t, idx_t, kSpatialDim, padding_mode>(
                                         i_shifted, sizeH, strideH, j_shifted, sizeW, strideW,
                                         k_shifted+1, sizeD, strideD, c, strideC,
                                         i_left_border, j_left_border, k_left_border,
                                         i_right_border, j_right_border, k_right_border,
                                         array, zero_point);
        output_values[5] = get_shifted_value<scalar_t, idx_t, kSpatialDim, padding_mode>(
                                         i_shifted+1, sizeH, strideH, j_shifted, sizeW, strideW,
                                         k_shifted+1, sizeD, strideD, c, strideC,
                                         i_left_border, j_left_border, k_left_border,
                                         i_right_border, j_right_border, k_right_border,
                                         array, zero_point);
        output_values[6] = get_shifted_value<scalar_t, idx_t, kSpatialDim, padding_mode>(
                                         i_shifted, sizeH, strideH, j_shifted+1, sizeW, strideW,
                                         k_shifted+1, sizeD, strideD, c, strideC,
                                         i_left_border, j_left_border, k_left_border,
                                         i_right_border, j_right_border, k_right_border,
                                         array, zero_point);
        output_values[7] = get_shifted_value<scalar_t, idx_t, kSpatialDim, padding_mode>(
                                         i_shifted+1, sizeH, strideH, j_shifted+1, sizeW, strideW,
                                         k_shifted+1, sizeD, strideD, c, strideC,
                                         i_left_border, j_left_border, k_left_border,
                                         i_right_border, j_right_border, k_right_border,
                                         array, zero_point);
    }                           
}


template <typename scalar_t, typename idx_t, int kSpatialDim = 1>
API_INLINE scalar_t compute_interpolated(scalar_t* v, scalar_t diff_shiftH, scalar_t diff_shiftW, scalar_t diff_shiftD){
    scalar_t res;
    switch (kSpatialDim){        
        case 3:
            res = interp3D(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7],
                           diff_shiftH, diff_shiftW, diff_shiftD);
            break;
        case 2:
            res = interp2D(v[0], v[1], v[2], v[3], 
                           diff_shiftH, diff_shiftW);
            break;
        case 1:
            res = interp1D(v[0], v[1], diff_shiftH);
            break;
    }
    return res;
}

template <typename scalar_t, typename idx_t, int kSpatialDim = 1>
API_INLINE void compute_weight_gradients(scalar_t* v, scalar_t diff_shiftH, scalar_t diff_shiftW, scalar_t diff_shiftD,
                                         scalar_t* output_grad){
    switch (kSpatialDim){        
        case 3:
            output_grad[0]=interp3D_dx(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7],
                                    diff_shiftW, diff_shiftD);
            output_grad[1]=interp3D_dy(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7],
                                    diff_shiftH, diff_shiftD);
            output_grad[2]=interp3D_dz(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7],
                                    diff_shiftH, diff_shiftW);
            break;
        case 2:
            output_grad[0]=interp2D_dx(v[0], v[1], v[2], v[3], 
                                    diff_shiftW);
            output_grad[1]=interp2D_dy(v[0], v[1], v[2], v[3], 
                                    diff_shiftH);
            break;
        case 1:
            output_grad[0]=interp1D_dx(v[0], v[1]);
            break;
    }
}

template <typename scalar_t, typename idx_t,
          int kSpatialDim = 1,
          BIPadding padding_mode = BIPadding::Zeros,
          bool active = false>
API_INLINE void shift_forward_kernel_nchwd(scalar_t* input, scalar_t* output,
                                           idx_t* weights, scalar_t* dweights,
                                           idx_t n, idx_t c, idx_t i, idx_t j, idx_t k,
                                           idx_t sizeH, idx_t sizeW, idx_t sizeD,
                                           idx_t input_sN, idx_t input_sC, idx_t input_sH, idx_t input_sW, idx_t input_sD,
                                           idx_t output_sN, idx_t output_sC, idx_t output_sH, idx_t output_sW, idx_t output_sD,
                                           idx_t weights_sC, idx_t weights_sS, idx_t dweights_sC, idx_t dweights_sS,
                                           idx_t i_left_border, idx_t j_left_border, idx_t k_left_border,
                                           idx_t i_right_border, idx_t j_right_border, idx_t k_right_border){
    // i, j, k - dispatch - 0
    scalar_t *input_NC = input + n*input_sN + c*input_sC;
    bool pass_cond = (i >= i_left_border)&&(i < i_right_border);
    if (kSpatialDim > 1) { pass_cond *= (j >= j_left_border)&&(j < j_right_border);}
    if (kSpatialDim > 2) { pass_cond *= (k >= k_left_border)&&(k < k_right_border);}
    if (pass_cond){
        idx_t oi = i - i_left_border;
        idx_t oj = j;
        idx_t ok = k;
        scalar_t val;
        idx_t si = i-*(weights+c*weights_sC);
        idx_t sj = j;
        idx_t sk = k;
        if (active)
        {   
            scalar_t di = *(dweights + c*dweights_sC);
            scalar_t dj = zp;
            scalar_t dk = zp;
            if (kSpatialDim > 1) { 
                oj -= j_left_border;
                sj -= *(weights+c*weights_sC+weights_sS);
                dj = *(dweights + c*dweights_sC + dweights_sS);
            }
            if (kSpatialDim > 2) { 
                ok -= k_left_border;
                sk -= *(weights+c*weights_sC+2*weights_sS);
                dk = *(dweights + c*dweights_sC + 2*dweights_sS); 
            }
 
            scalar_t _vals_array[8] = {zp, zp, zp, zp, zp, zp, zp, zp};
            get_shifted_values<scalar_t,idx_t,kSpatialDim, padding_mode>(
                                               si, sizeH, input_sH,
                                               sj, sizeW, input_sW,
                                               sk, sizeD, input_sD,
                                               0, 0, 0, 0, 0, sizeH, sizeW, sizeD,
                                               input_NC, zp, _vals_array);
            val = compute_interpolated<scalar_t,idx_t,kSpatialDim>(_vals_array, di, dj, dk);
        }
        else {
            if (kSpatialDim > 1) { 
                oj -= j_left_border;
                sj -= *(weights+c*weights_sC+weights_sS);
            }
            if (kSpatialDim > 2) { 
                ok -= k_left_border;
                sk -= *(weights+c*weights_sC+2*weights_sS);
            }
            val = get_shifted_value<scalar_t,idx_t,kSpatialDim>(
                                                si, sizeH, input_sH,
                                                sj, sizeW, input_sW,
                                                sk, sizeD, input_sD,
                                                0, 0, 0, 0, 0, sizeH, sizeW, sizeD,
                                                input_NC, zp, padding_mode);  
       }
       scalar_t *output_NCHWD = output + n*output_sN + c*output_sC + oi*output_sH + oj*output_sW + ok*output_sD;
       *output_NCHWD = val;
    }
}

template <typename scalar_t, typename idx_t,
          int kSpatialDim = 1,
          BIPadding padding_mode = BIPadding::Zeros,
          bool active = false>
API_INLINE void shift_backward_kernel_nchwd(scalar_t* input_grad, scalar_t* input,  scalar_t* output_grad,
                                            idx_t* weights, scalar_t* dweights, scalar_t* weights_grad,
                                            idx_t n, idx_t c, idx_t i, idx_t j, idx_t k,
                                            idx_t sizeH, idx_t sizeW, idx_t sizeD,
                                            idx_t input_grad_sN, idx_t input_grad_sC, idx_t input_grad_sH, idx_t input_grad_sW, idx_t input_grad_sD,
                                            idx_t input_sN, idx_t input_sC, idx_t input_sH, idx_t input_sW, idx_t input_sD,
                                            idx_t output_grad_sN, idx_t output_grad_sC, idx_t output_grad_sH, idx_t output_grad_sW, idx_t output_grad_sD,
                                            idx_t weights_sC, idx_t weights_sS, idx_t dweights_sC, idx_t dweights_sS, idx_t weights_grad_sC, idx_t weights_grad_sS,
                                            idx_t i_left_border, idx_t j_left_border, idx_t k_left_border,
                                            idx_t i_right_border, idx_t j_right_border, idx_t k_right_border){
    scalar_t *input_grad_NC = input_grad + n*input_grad_sN + c*input_grad_sC;
    scalar_t *input_NC = input + n*input_sN + c*input_sC;
    idx_t oi = i+i_left_border;
    idx_t oj = j;
    idx_t ok = k;                             
    scalar_t zp = static_cast<scalar_t>(0);
    scalar_t _vals_array[8] = {zp, zp, zp, zp, zp, zp, zp, zp};
    idx_t shifti = *(weights+c*weights_sC);
    idx_t shiftj = 0;
    idx_t shiftk = 0;
    scalar_t di = *(dweights + c*dweights_sC);
    scalar_t dj = zp;
    scalar_t dk = zp;
    idx_t osi = oi - shifti;
    idx_t osj = oj;
    idx_t osk = ok;
    scalar_t input_grad_NCHWD_val = input_grad_NC[i*input_grad_sH + j*input_grad_sW + k*input_grad_sD];
    if (kSpatialDim > 1){
        oj += j_left_border;
        shiftj = *(weights+c*weights_sC+weights_sS)     
        dj = *(dweights + c*dweights_sC + dweights_sS);
        osj -= shiftj;
    }
    if (kSpatialDim > 2){
        ok += k_left_border;
        shiftk = *(weights + c*weights_sC + 2*weights_sS);        
        dk = *(dweights + c*dweights_sC + 2*dweights_sS);
        osk -= shiftk;
    }
    if (active)
    {
        idx_t si = i-shifti;
        idx_t sj = 0;
        idx_t sk = 0;
        if (kSpatialDim > 1){ sj -= shiftj; }
        if (kSpatialDim > 2){ sk -= shiftk; }
        get_shifted_values<scalar_t,idx_t,kSpatialDim,padding_mode>(
                                        si, sizeH, input_grad_sH,
                                        sj, sizeW, input_grad_sW,
                                        sk, sizeD, input_grad_sD,
                                        0, 0, i_left_border, j_left_border, k_left_border,
                                        i_right_border, j_right_border, k_right_border,
                                        input_grad_NC, zp,  _vals_array);
        scalar_t *output_grad_NCHWD = output_grad + n*output_grad_sN + c*output_grad_sC +
                                      oi*output_grad_sH + oj*output_grad_sW + ok*output_grad_sD; 
        *output_grad_NCHWD = compute_interpolated<scalar_t,idx_t,kSpatialDim>(
                                        _vals_array, di, dj, dk);
    } 
    else { 
        idx_t rsi = i+shifti;
        idx_t rsj = 0;
        idx_t rsk = 0;
        if (kSpatialDim > 1){ rsj += shiftj; }
        if (kSpatialDim > 2){ rsk += shiftk; }
        scalar_t *output_grad_NCHWD = output_grad + n*output_grad_sN + c*output_grad_sC +
                                      oi*output_grad_sH + oj*output_grad_sW + ok*output_grad_sD;                                                      
        *output_grad_NCHWD = get_shifted_value<scalar_t,idx_t,kSpatialDim,padding_mode>(
                                        rsi, sizeH, input_grad_sH,
                                        rsj, sizeW, input_grad_sW,
                                        rsk, sizeD, input_grad_sD,
                                        0, 0, i_left_border, j_left_border, k_left_border,
                                        i_right_border, j_right_border, k_right_border,
                                        input_grad_NC, zp);
    }
    get_shifted_values<scalar_t,idx_t,kSpatialDim,padding_mode>(
                                        osi, sizeH, input_sH,
                                        osj, sizeW, input_sW,
                                        osk, sizeD, input_sD,
                                        0, 0, 0, 0, 0, sizeH, sizeW, sizeD,                                       
                                        input_NC, zp, _vals_array);
    scalar_t _new_weights_grad[3] = {zp, zp, zp};                                   
    compute_weight_gradients<scalar_t,idx_t, kSpatialDim>(
                                _vals_array, di, dj, dk, _new_weights_grad);
    ADD((weights_grad + c*weights_grad_sC),(input_grad_NCHWD_val * _new_weights_grad[0]));
    if (kSpatialDim > 1){ADD((weights_grad + c*weights_grad_sC + weights_grad_sS),(input_grad_NCHWD_val * _new_weights_grad[1]));}
    if (kSpatialDim > 2){ADD((weights_grad + c*weights_grad_sC + 2*weights_grad_sS),(input_grad_NCHWD_val * _new_weights_grad[2]));}
}


template <typename scalar_t, typename idx_t,
          int kSpatialDim = 1,
          BIPadding padding_mode = BIPadding::Zeros,
          bool active = false>
API_INLINE void shift_forward_kernel_nhwdc(scalar_t* input, scalar_t* output, 
                                           idx_t* weights, scalar_t* dweights,
                                           idx_t n, idx_t i, idx_t j, idx_t k,
                                           idx_t sizeC, idx_t sizeH, idx_t sizeW, idx_t sizeD,
                                           idx_t input_sN, idx_t input_sC, idx_t input_sH, idx_t input_sW, idx_t input_sD,
                                           idx_t output_sN, idx_t output_sC, idx_t output_sH, idx_t output_sW, idx_t output_sD,
                                           idx_t weights_sC, idx_t weights_sS, idx_t dweights_sC, idx_t dweights_sS,
                                           idx_t i_left_border, idx_t j_left_border, idx_t k_left_border,
                                           idx_t i_right_border, idx_t j_right_border, idx_t k_right_border){
    scalar_t *input_N = input + n*input_sN;
    bool pass_cond = (i >= i_left_border)&&(i < i_right_border);
    if (kSpatialDim > 1) { pass_cond *= (j >= j_left_border)&&(j < j_right_border);}
    if (kSpatialDim > 2) { pass_cond *= (k >= k_left_border)&&(k < k_right_border);}
    if (pass_cond){
        idx_t oi = i - i_left_border;
        idx_t oj = j;
        idx_t ok = k;
        if (kSpatialDim > 1) { oj -= j_left_border; };
        if (kSpatialDim > 2) { ok -= k_left_border; };
        scalar_t zp = static_cast<scalar_t>(0);
        scalar_t val;
        idx_t si = i;
        idx_t sj = j;
        idx_t sk = k;
        scalar_t di = zp;
        scalar_t dj = zp;
        scalar_t dk = zp;
        scalar_t *output_NHWD = output + n*output_sN + oi*output_sH + oj*output_sW + ok*output_sD;
        for (idx_t c = 0; c < sizeC; c++)
        {
            si -= *(weights+c*weights_sC);
            if (kSpatialDim > 1) { sj -= *(weights+weights_sS+c*weights_sC); }
            if (kSpatialDim > 2) { sk -= *(weights+2*weights_sS+c*weights_sC); }
            if (active)
            {
                di = *(dweights + c*dweights_sC);
                if (kSpatialDim > 1) { dj = *(dweights+dweights_sS+c*dweights_sC); }
                if (kSpatialDim > 2) { dk = *(dweights+2*dweights_sS+c*dweights_sC); }
                // define array here to avoid unnessary warnings, Hope the compiler can optimize it itself
                scalar_t _vals_array[8] = {zp, zp, zp, zp, zp, zp, zp, zp};
                get_shifted_values<scalar_t,idx_t,kSpatialDim,padding_mode>(
                                                si, sizeH, input_sH,
                                                sj, sizeW, input_sW,
                                                sk, sizeD, input_sD,
                                                c, input_sC, 
                                                0, 0, 0, sizeH, sizeW, sizeD,
                                                input_N, zp, _vals_array);
                val = compute_interpolated<scalar_t,idx_t,kSpatialDim>(_vals_array, di, dj, dk);
            }
            else {   
                val = get_shifted_value<scalar_t,idx_t,kSpatialDim,padding_mode>(
                                                si, sizeH, input_sH,
                                                sj, sizeW, input_sW,
                                                sk, sizeD, input_sD,
                                                c, input_sC, 
                                                0, 0, 0, sizeH, sizeW, sizeD,
                                                input_N, zp);
            }
            output_NHWD[c*output_sC] = val;
        }
    }
}

template <typename scalar_t, typename idx_t,
          int kSpatialDim = 1,
          BIPadding padding_mode = BIPadding::Zeros,
          bool active = false>
API_INLINE void shift_backward_kernel_nhwdc(scalar_t* input_grad, scalar_t* input,  scalar_t* output_grad,
                                            idx_t* weights, scalar_t* dweights, scalar_t* weights_grad,
                                            idx_t n, idx_t i, idx_t j, idx_t k,
                                            idx_t sizeC, idx_t sizeH, idx_t sizeW, idx_t sizeD,
                                            idx_t input_grad_sN, idx_t input_grad_sC, idx_t input_grad_sH, idx_t input_grad_sW, idx_t input_grad_sD,
                                            idx_t input_sN, idx_t input_sC, idx_t input_sH, idx_t input_sW, idx_t input_sD,
                                            idx_t output_grad_sN, idx_t output_grad_sC, idx_t output_grad_sH, idx_t output_grad_sW, idx_t output_grad_sD,
                                            idx_t weights_sC, idx_t weights_sS, idx_t dweights_sC, idx_t dweights_sS, idx_t weights_grad_sC, idx_t weights_grad_sS,
                                            idx_t i_left_border, idx_t j_left_border, idx_t k_left_border,
                                            idx_t i_right_border, idx_t j_right_border, idx_t k_right_border){
    scalar_t *input_grad_N = input_grad + n*input_grad_sN;
    scalar_t *input_N = input + n*input_sN;
    idx_t oi = i+i_left_border;
    idx_t oj = j
    idx_t ok = k;
    if (kSpatialDim > 1) { oj -= j_left_border; };
    if (kSpatialDim > 2) { ok -= k_left_border; };
    scalar_t *output_grad_NHWD= output_grad + n*output_grad_sN + oi*output_grad_sH + oj*output_grad_sW + ok*output_grad_sD;         
    scalar_t *input_grad_NHWD = input_grad_N + i*input_grad_sH + j*input_grad_sW + k*input_grad_sD;
    scalar_t input_grad_NHWDC_val; 
    scalar_t zp = static_cast<scalar_t>(0);
    idx_t shifti = 0;
    idx_t shiftj = 0;
    idx_t shiftk = 0;
    scalar_t di = zp;
    scalar_t dj = zp;
    scalar_t dk = zp;
    idx_t osi = oi;
    idx_t osj = oj;
    idx_t osk = ok;
    idx_t si = i;
    idx_t sj = j;
    idx_t sk = k;
    idx_t rsi = i;
    idx_t rsj = j;
    idx_t rsk = k;

    scalar_t _vals_array[8] = {zp, zp, zp, zp, zp, zp, zp, zp};
    scalar_t _new_weights_grad[3] = {zp, zp, zp};
    for (idx_t c = 0; c < sizeC; c++)
    {
        shifti = *(weights+c*weights_sC);
        di = *(dweights+c*dweights_sC)
        osi -= shifti;
        if (kSpatialDim > 1) {
            shiftj = *(weights+weights_sS+c*weights_sC);
            dj = *(dweights+dweights_sS+c*dweights_sC);
            osj -= shiftj;
        }
        if (kSpatialDim > 2) {
            shiftk = *(weights+2*weights_sS+c*weights_sC);
            dk = *(dweights+2*dweights_sS+c*dweights_sC);
            osk -= shiftk;
        }
        if (active)
        {   
            si -= shifti;
            if (kSpatialDim > 1) {  sj -= shiftj; }
            if (kSpatialDim > 2) {  sk -= shiftk; }
            get_shifted_values<scalar_t,idx_t,kSpatialDim,padding_mode>(
                                        si, sizeH, input_grad_sH,
                                        sj, sizeW, input_grad_sW,
                                        sk, sizeD, input_grad_sD,
                                        c, input_grad_sC, i_left_border, j_left_border, k_left_border,
                                        i_right_border, j_right_border, k_right_border,    
                                        input_grad_N, zp, _vals_array);
            *(output_grad_NHWD+c*output_grad_sC) = compute_interpolated<scalar_t,idx_t,kSpatialDim>(
                                        _vals_array, di, dj, dk);
        }
        else {
            rsi += shifti;
            if (kSpatialDim > 1) {  rsj += shiftj; }
            if (kSpatialDim > 2) {  rsk += shiftk; }
            *(output_grad_NHWD+c*output_grad_sC) =  get_shifted_value<scalar_t,idx_t,kSpatialDim,padding_mode>(
                                        rsi, sizeH, input_grad_sH,
                                        rsj, sizeW, input_grad_sW,
                                        rsk, sizeD, input_grad_sD,
                                        c, input_grad_sC, i_left_border, j_left_border, k_left_border,
                                        i_right_border, j_right_border, k_right_border,
                                        input_grad_N, zp);
        }
        get_shifted_values<scalar_t,idx_t,kSpatialDim,padding_mode>(
                                        osi, sizeH, input_sH,
                                        osj, sizeW, input_sW,
                                        osk, sizeD, input_sD,
                                        c, input_sC, 0, 0, 0, sizeH, sizeW, sizeD,
                                        input_N, zp, _vals_array);
        compute_weight_gradients<scalar_t,idx_t,kSpatialDim>(_vals_array, di, dj, dk, _new_weights_grad);
        input_grad_NHWDC_val = input_grad_NHWD[c*input_grad_sC];
        ADD((weights_grad + c*weights_grad_sC),(input_grad_NHWDC_val * _new_weights_grad[0]));
        if (kSpatialDim > 1){ADD((weights_grad + weights_grad_sS + c*weights_grad_sC),(input_grad_NHWDC_val * _new_weights_grad[1]));}
        if (kSpatialDim > 2){ADD((weights_grad + 2*weights_grad_sS + c*weights_grad_sC),(input_grad_NHWDC_val * _new_weights_grad[2]));}
    }
}


/////////QUANTIZED

template <typename scalar_t, typename idx_t,
          int kSpatialDim = 1,
          BIPadding padding_mode = BIPadding::Zeros>
API_INLINE void shift_forward_kernel_nchwd_q(scalar_t* input, scalar_t* output,
                                             idx_t* weights,
                                             idx_t n, idx_t c, idx_t i, idx_t j, idx_t k,
                                             idx_t sizeH, idx_t sizeW, idx_t sizeD,
                                             idx_t input_sN, idx_t input_sC, idx_t input_sH, idx_t input_sW, idx_t input_sD,
                                             idx_t output_sN, idx_t output_sC, idx_t output_sH, idx_t output_sW, idx_t output_sD,
                                             idx_t weights_sC, idx_t weights_sS,
                                             idx_t i_left_border, idx_t j_left_border, idx_t k_left_border,
                                             idx_t i_right_border, idx_t j_right_border, idx_t k_right_border,
                                             scalar_t zero_point, idx_t weights_zero_point){
    scalar_t *input_NC = input + n*input_sN + c*input_sC;
    bool pass_cond = (i >= i_left_border)&&(i < i_right_border);
    if (kSpatialDim > 1) { pass_cond *= (j >= j_left_border)&&(j < j_right_border);}
    if (kSpatialDim > 2) { pass_cond *= (k >= k_left_border)&&(k < k_right_border);}
    if (pass_cond){
        idx_t oi = i - i_left_border;
        idx_t oj = j;
        idx_t ok = k;
        idx_t si = i - *(weights+c*weights_sC) + weights_zero_point;
        idx_t sj = j;
        idx_t sk = k;
        if (kSpatialDim > 1){
            oj -= j_left_border;
            sj -= (*(weights+c*weights_sC+weights_sS) - weights_zero_point);
        }
        if (kSpatialDim > 2){
            ok -= k_left_border;
            sk -= (*(weights+c*weights_sC+2*weights_sS) - weights_zero_point);
        }
        scalar_t *output_NCHWD= output + n*output_sN + c*output_sC + oi*output_sH + oj*output_sW + ok*output_sD;
        *output_NCHWD = get_shifted_value<scalar_t,idx_t,kSpatialDim,padding_mode>(
                                            si, sizeH, input_sH,
                                            sj, sizeW, input_sW,
                                            sk, sizeD, input_sD,
                                            0, 0, 0, 0, 0, sizeH, sizeW, sizeD,
                                            input_NC, zero_point);
    }
}


template <typename scalar_t, typename idx_t,
          int kSpatialDim = 1,
          BIPadding padding_mode = BIPadding::Zeros>
API_INLINE void shift_forward_kernel_nhwdc_q(scalar_t* input, scalar_t* output, 
                                             idx_t* weights,
                                             idx_t n, idx_t i, idx_t j, idx_t k,
                                             idx_t sizeC, idx_t sizeH, idx_t sizeW, idx_t sizeD,
                                             idx_t input_sN, idx_t input_sC, idx_t input_sH, idx_t input_sW, idx_t input_sD,
                                             idx_t output_sN, idx_t output_sC, idx_t output_sH, idx_t output_sW, idx_t output_sD,
                                             idx_t weights_sC, idx_t weights_sS,
                                             idx_t i_left_border, idx_t j_left_border, idx_t k_left_border,
                                             idx_t i_right_border, idx_t j_right_border, idx_t k_right_border,
                                             scalar_t zero_point, idx_t weights_zero_point, BIPadding padding_mode){
    scalar_t *input_N = input + n*input_sN;
    bool pass_cond = (i >= i_left_border)&&(i < i_right_border);
    if (kSpatialDim > 1) { pass_cond *= (j >= j_left_border)&&(j < j_right_border);}
    if (kSpatialDim > 2) { pass_cond *= (k >= k_left_border)&&(k < k_right_border);}
    if (pass_cond){
        idx_t oi = i - i_left_border;
        idx_t oj = j;
        idx_t ok = k;
        idx_t si = i;
        idx_t sj = j;
        idx_t sk = k;
        if (kSpatialDim > 1){ oj -= j_left_border; }
        if (kSpatialDim > 2){ ok -= k_left_border; }

        scalar_t *output_NHWD = output + n*output_sN + oi*output_sH + oj*output_sW + ok*output_sD;
        for (idx_t c = 0; c < sizeC; c++)
        {
            si -= (*(weights+c*weights_sC) - weights_zero_point);
            if (kSpatialDim > 1){ sj -= (*(weights+weights_sS+c*weights_sC) - weights_zero_point); }
            if (kSpatialDim > 2){ sk -= (*(weights+2*weights_sS+c*weights_sC) - weights_zero_point); }
             output_NHWD[c*output_sC] = get_shifted_value<scalar_t,idx_t,kSpatialDim,padding_mode>(
                                                si, sizeH, input_sH,
                                                sj, sizeW, input_sW,
                                                sk, sizeD, input_sD,
                                                c, input_sC, 0, 0, 0, sizeH, sizeW, sizeD,
                                                input_N, zero_point);
         }
    }
}

