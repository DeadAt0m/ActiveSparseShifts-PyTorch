#include "../global_scope.h"
#include "interpolation.h"


enum class BIPadding {Zeros, Border, Periodic, Reflect, Symmetric};

template<typename T>
API_INLINE T mod(const T a, const T b){return (b + (a % b)) % b;}

template<typename idx_t, BIPadding padding_mode = BIPadding::Zeros>
API_INLINE idx_t infer_index(const idx_t index, const idx_t len){    
    idx_t odd_seq;
    idx_t out_index;
    switch (padding_mode){        
        case BIPadding::Zeros:
            out_index = -MAX(static_cast<idx_t>(1),static_cast<idx_t>(2*MAX(index-len,static_cast<idx_t>(-1))+1))*
                        MAX(index,static_cast<idx_t>(-1));
            break;
        case BIPadding::Border:
            out_index = MIN(len-1,MAX(index,static_cast<idx_t>(0)));
            break;
        case BIPadding::Periodic:
            out_index = mod<idx_t>(index, len);
            break;
        case BIPadding::Reflect:
            odd_seq = ((ABS(index)+MAX(static_cast<idx_t>(-1),MIN(index,static_cast<idx_t>(0))))/(len-1) - 
                      MAX(static_cast<idx_t>(-1),MIN(index,static_cast<idx_t>(0)))) & 1;
            out_index = mod<idx_t>(index, len-1)*(1-odd_seq) +
                        odd_seq*(len-1-mod<idx_t>(index, len-1));
            break;
        case BIPadding::Symmetric:       
            odd_seq = ((ABS(index)+MAX(static_cast<idx_t>(-1),MIN(index,static_cast<idx_t>(0))))/len - 
                      MAX(static_cast<idx_t>(-1),MIN(index,static_cast<idx_t>(0)))) & 1;
            out_index = mod<idx_t>(index, len)*(1-odd_seq) + 
                        odd_seq*(len-1-mod<idx_t>(index, len));
            break;
    }
    return out_index;
}




template<typename scalar_t, typename idx_t, 
         int kSpatialDim = 1,
         BIPadding padding_mode = BIPadding::Zeros>
API_INLINE scalar_t get_shifted_value(const idx_t i_shifted, const idx_t sizeH, const idx_t strideH,
                                      const idx_t j_shifted, const idx_t sizeW, const idx_t strideW,
                                      const idx_t k_shifted, const idx_t sizeD, const idx_t strideD,
                                      const idx_t c, const idx_t strideC, const idx_t out_passcond,
                                      const scalar_t* const array, const scalar_t zero_point){
    const idx_t tidx_i = (sizeH==1)?0:infer_index<idx_t,padding_mode>(i_shifted, sizeH);
    const idx_t pass_cond_i = 1+MAX(static_cast<idx_t>(-1),MIN(tidx_i,static_cast<idx_t>(0)));
//     const idx_t pass_cond_i = ABS(MAX(tidx_i,0)*MAX(tidx_i-i_left_border+1,0)*MAX(i_right_border-tidx_i,0));
    const idx_t isH = tidx_i*strideH*pass_cond_i;

    const idx_t tidx_j = (kSpatialDim > 1)?((sizeW==1)?0:infer_index<idx_t,padding_mode>(j_shifted, sizeW)):0;
    const idx_t pass_cond_j = (kSpatialDim > 1)?(1+MAX(static_cast<idx_t>(-1),MIN(tidx_j,static_cast<idx_t>(0))))*pass_cond_i:pass_cond_i;                               
    const idx_t isW = (kSpatialDim > 1)?tidx_j*strideW*pass_cond_j:0;       
                                                 
    const idx_t tidx_k = (kSpatialDim > 2)?((sizeD==1)?0:infer_index<idx_t,padding_mode>(k_shifted, sizeD)):0;
    const idx_t pass_cond_k = (kSpatialDim > 2)?(1+MAX(static_cast<idx_t>(-1),MIN(tidx_k,static_cast<idx_t>(0))))*pass_cond_j:pass_cond_j;                               
    const idx_t isD = (kSpatialDim > 2)?tidx_k*strideD*pass_cond_k:0;              
                   
    const idx_t pass_cond = pass_cond_k*out_passcond;

    //This dirty hack intedens for using with quantized version, because +/*/- operators not overloaded for Pytorch Qtypes
    #if defined(_SHIFTS_QCPU)
        return static_cast<bool>(pass_cond)?array[isH+isW+isD+c*strideC]:zero_point;
    #else
        return static_cast<scalar_t>(pass_cond)*array[isH+isW+isD+c*strideC]+static_cast<scalar_t>(1-pass_cond)*zero_point;
    #endif         
}



template<typename scalar_t, typename idx_t,
         int kSpatialDim = 1,
         BIPadding padding_mode = BIPadding::Zeros>
API_INLINE void get_shifted_values(const idx_t i_shifted, const idx_t sizeH, const idx_t strideH,
                                   const idx_t j_shifted, const idx_t sizeW, const idx_t strideW,
                                   const idx_t k_shifted, const idx_t sizeD, const idx_t strideD,
                                   const idx_t c, const idx_t strideC, const idx_t out_passcond,
                                   const scalar_t* const array, const scalar_t zero_point,
                                   scalar_t* const output_values){
    output_values[0] = get_shifted_value<scalar_t, idx_t, kSpatialDim, padding_mode>(
                                         i_shifted, sizeH, strideH, j_shifted, sizeW, strideW,
                                         k_shifted, sizeD, strideD, c, strideC,
                                         out_passcond, array, zero_point);
    output_values[1] = get_shifted_value<scalar_t, idx_t, kSpatialDim, padding_mode>(
                                         i_shifted+1, sizeH, strideH, j_shifted, sizeW, strideW,
                                         k_shifted, sizeD, strideD, c, strideC, 
                                         out_passcond, array, zero_point);
    if (kSpatialDim > 1){
        output_values[2] = get_shifted_value<scalar_t, idx_t, kSpatialDim, padding_mode>(
                                         i_shifted, sizeH, strideH, j_shifted+1, sizeW, strideW,
                                         k_shifted, sizeD, strideD, c, strideC,
                                         out_passcond, array, zero_point);
        output_values[3] = get_shifted_value<scalar_t, idx_t, kSpatialDim, padding_mode>(
                                         i_shifted+1, sizeH, strideH, j_shifted+1, sizeW, strideW,
                                         k_shifted, sizeD, strideD, c, strideC,
                                         out_passcond, array, zero_point);                           
    }
    if (kSpatialDim > 2){
        output_values[4] = get_shifted_value<scalar_t, idx_t, kSpatialDim, padding_mode>(
                                         i_shifted, sizeH, strideH, j_shifted, sizeW, strideW,
                                         k_shifted+1, sizeD, strideD, c, strideC,
                                         out_passcond, array, zero_point);
        output_values[5] = get_shifted_value<scalar_t, idx_t, kSpatialDim, padding_mode>(
                                         i_shifted+1, sizeH, strideH, j_shifted, sizeW, strideW,
                                         k_shifted+1, sizeD, strideD, c, strideC,
                                         out_passcond, array, zero_point);
        output_values[6] = get_shifted_value<scalar_t, idx_t, kSpatialDim, padding_mode>(
                                         i_shifted, sizeH, strideH, j_shifted+1, sizeW, strideW,
                                         k_shifted+1, sizeD, strideD, c, strideC,
                                         out_passcond, array, zero_point);
        output_values[7] = get_shifted_value<scalar_t, idx_t, kSpatialDim, padding_mode>(
                                         i_shifted+1, sizeH, strideH, j_shifted+1, sizeW, strideW,
                                         k_shifted+1, sizeD, strideD, c, strideC,
                                         out_passcond, array, zero_point);
    }                           
}

template <typename scalar_t, bool reverse>
API_INLINE scalar_t rev_shift(const scalar_t diff_shift){
    return (reverse)?(static_cast<scalar_t>(1)-diff_shift):diff_shift;
}

template <typename scalar_t, typename idx_t, int kSpatialDim = 1, bool reverse = false>
API_INLINE scalar_t compute_interpolated(const scalar_t* const v, const scalar_t diff_shiftH, 
                                         const scalar_t diff_shiftW, const scalar_t diff_shiftD,
                                         const idx_t pass_cond, const scalar_t zp){
    scalar_t res;
    switch (kSpatialDim){        
        case 3:
            res = interp3D(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7],
                           rev_shift<scalar_t,reverse>(diff_shiftH), 
                           rev_shift<scalar_t,reverse>(diff_shiftW),
                           rev_shift<scalar_t,reverse>(diff_shiftD));
            break;
        case 2:
            res = interp2D(v[0], v[1], v[2], v[3], 
                           rev_shift<scalar_t,reverse>(diff_shiftH), 
                           rev_shift<scalar_t,reverse>(diff_shiftW));
            break;
        case 1:
            res = interp1D(v[0], v[1], rev_shift<scalar_t,reverse>(diff_shiftH));
            break;
    }
    return static_cast<scalar_t>(pass_cond)*res+static_cast<scalar_t>(1-pass_cond)*zp;
}

template <typename scalar_t, typename idx_t, int kSpatialDim = 1>
API_INLINE void compute_weight_gradients(const scalar_t* const v, const scalar_t diff_shiftH, const scalar_t diff_shiftW, const scalar_t diff_shiftD,
                                         scalar_t* const output_grad){
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
API_INLINE void shift_forward_kernel_nchwd(const scalar_t* const input, scalar_t* const output,
                                           const idx_t* const weights, const scalar_t* const dweights,
                                           const idx_t n, const idx_t c, const idx_t i, const idx_t j, const idx_t k,
                                           const idx_t sizeH, const idx_t sizeW, const idx_t sizeD,
                                           const idx_t input_sN, const idx_t input_sC, const idx_t input_sH, const idx_t input_sW, const idx_t input_sD,
                                           const idx_t output_sN, const idx_t output_sC, const idx_t output_sH, const idx_t output_sW, const idx_t output_sD,
                                           const idx_t weights_sC, const idx_t weights_sS, const idx_t dweights_sC, const idx_t dweights_sS,
                                           const idx_t i_left_border, const idx_t j_left_border, const idx_t k_left_border,
                                           const idx_t i_right_border, const idx_t j_right_border, const idx_t k_right_border){
    const scalar_t* const input_NC = input + n*input_sN + c*input_sC;
    const scalar_t zp = static_cast<scalar_t>(0);
              
    const idx_t oi = i - i_left_border;
    const idx_t oj = (kSpatialDim > 1) ? (j - j_left_border) : j; 
    const idx_t ok = (kSpatialDim > 2) ? (k - k_left_border) : k;
    
    const idx_t si = i - *(weights+c*weights_sC);
    const idx_t sj = (kSpatialDim > 1) ? (j - *(weights+c*weights_sC+weights_sS)) : j;
    const idx_t sk = (kSpatialDim > 2) ? (k - *(weights+c*weights_sC+2*weights_sS)) : k;       
    
    const scalar_t di = *(dweights + c*dweights_sC);
    const scalar_t dj = (kSpatialDim > 1) ? *(dweights + c*dweights_sC + dweights_sS) : zp;
    const scalar_t dk = (kSpatialDim > 2) ?  *(dweights + c*dweights_sC + 2*dweights_sS): zp; 
              
    const idx_t pass_cond_i = (1+MAX(static_cast<idx_t>(-1),MIN(i_right_border-i-1,static_cast<idx_t>(0))))*
                              (1+MAX(static_cast<idx_t>(-1),MIN(i-i_left_border, static_cast<idx_t>(0))));
    const idx_t pass_cond_j = (kSpatialDim > 1)?(1+MAX(static_cast<idx_t>(-1),MIN(j_right_border-j-1,static_cast<idx_t>(0))))*
                                                (1+MAX(static_cast<idx_t>(-1),MIN(j-j_left_border, static_cast<idx_t>(0)))):1;
    const idx_t pass_cond_k = (kSpatialDim > 2)?(1+MAX(static_cast<idx_t>(-1),MIN(k_right_border-k-1,static_cast<idx_t>(0))))*
                                                (1+MAX(static_cast<idx_t>(-1),MIN(k-k_left_border, static_cast<idx_t>(0)))):1;
    const idx_t pass_cond = pass_cond_i * pass_cond_j * pass_cond_k;
    
    scalar_t val;
    scalar_t* output_NCHWD = output + (n*output_sN + c*output_sC + oi*output_sH + oj*output_sW + ok*output_sD)*pass_cond;
    
    if (active)
    {   
        scalar_t _vals_array[8] = {zp, zp, zp, zp, zp, zp, zp, zp};
        get_shifted_values<scalar_t,idx_t,kSpatialDim, padding_mode>(
                                          si, sizeH, input_sH,
                                          sj, sizeW, input_sW,
                                          sk, sizeD, input_sD,
                                          0, 0, pass_cond, 
                                          input_NC, zp, _vals_array);
        val = compute_interpolated<scalar_t,idx_t,kSpatialDim,false>(_vals_array, di, dj, dk, pass_cond, zp);
    }
    else {
        val = get_shifted_value<scalar_t,idx_t,kSpatialDim,padding_mode>(
                                                si, sizeH, input_sH,
                                                sj, sizeW, input_sW,
                                                sk, sizeD, input_sD,
                                                0, 0, pass_cond, 
                                                input_NC, zp);  
    }
    *output_NCHWD = static_cast<scalar_t>(pass_cond)*val+static_cast<scalar_t>(1-pass_cond)*(*output_NCHWD);
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
    // i,j,k - from input
    const scalar_t* const input_grad_NC = input_grad + n*input_grad_sN + c*input_grad_sC;
    const scalar_t* const input_NC = input + n*input_sN + c*input_sC;
    const scalar_t zp = static_cast<scalar_t>(0);
    scalar_t _vals_array[8] = {zp, zp, zp, zp, zp, zp, zp, zp};
    scalar_t _new_weights_grad[3] = {zp, zp, zp};
    
    const idx_t shifti = *(weights+c*weights_sC);
    const idx_t shiftj = (kSpatialDim > 1)?*(weights+c*weights_sC + weights_sS):0;
    const idx_t shiftk = (kSpatialDim > 2)?*(weights + c*weights_sC + 2*weights_sS):0;
              
    const scalar_t di = *(dweights + c*dweights_sC);
    const scalar_t dj = (kSpatialDim > 1)?*(dweights + c*dweights_sC + dweights_sS):zp;
    const scalar_t dk = (kSpatialDim > 2)?*(dweights + c*dweights_sC + 2*dweights_sS):zp;

    const idx_t si = i - *(weights+c*weights_sC);
    const idx_t sj = (kSpatialDim > 1) ? (j - *(weights+c*weights_sC+weights_sS)) : j;
    const idx_t sk = (kSpatialDim > 2) ? (k - *(weights+c*weights_sC+2*weights_sS)) : k;       
    
    const idx_t pass_cond_i = (1+MAX(static_cast<idx_t>(-1),MIN(i_right_border-i-1,static_cast<idx_t>(0))))*
                              (1+MAX(static_cast<idx_t>(-1),MIN(i-i_left_border, static_cast<idx_t>(0))));
    const idx_t pass_cond_j = (kSpatialDim > 1)?(1+MAX(static_cast<idx_t>(-1),MIN(j_right_border-j-1,static_cast<idx_t>(0))))*
                                                (1+MAX(static_cast<idx_t>(-1),MIN(j-j_left_border, static_cast<idx_t>(0)))):1;
    const idx_t pass_cond_k = (kSpatialDim > 2)?(1+MAX(static_cast<idx_t>(-1),MIN(k_right_border-k-1,static_cast<idx_t>(0))))*
                                                (1+MAX(static_cast<idx_t>(-1),MIN(k-k_left_border, static_cast<idx_t>(0)))):1;
    const idx_t pass_cond = pass_cond_i * pass_cond_j * pass_cond_k;
    
    const idx_t oi = i - i_left_border;
    const idx_t oj = (kSpatialDim > 1) ? (j - j_left_border) : j; 
    const idx_t ok = (kSpatialDim > 2) ? (k - k_left_border) : k;    
    const scalar_t input_grad_NCHWD_val = input_grad_NC[pass_cond*(oi*input_grad_sH + oj*input_grad_sW + ok*input_grad_sD)];
    
    // weight gradients
    get_shifted_values<scalar_t,idx_t,kSpatialDim,padding_mode>(
                                        si, sizeH, input_sH,
                                        sj, sizeW, input_sW,
                                        sk, sizeD, input_sD,
                                        0, 0, pass_cond,                                       
                                        input_NC, zp, _vals_array);                         
    compute_weight_gradients<scalar_t,idx_t, kSpatialDim>(_vals_array, di, dj, dk, _new_weights_grad);    
    ADD((weights_grad + c*weights_grad_sC),(input_grad_NCHWD_val * _new_weights_grad[0]));
    if (kSpatialDim > 1){ADD((weights_grad + c*weights_grad_sC + weights_grad_sS),(input_grad_NCHWD_val * _new_weights_grad[1]));}
    if (kSpatialDim > 2){ADD((weights_grad + c*weights_grad_sC + 2*weights_grad_sS),(input_grad_NCHWD_val * _new_weights_grad[2]));}
    
    // input gradient
    scalar_t* output_grad_NCHWD = output_grad + n*output_grad_sN + c*output_grad_sC +
                                  i*output_grad_sH + j*output_grad_sW + k*output_grad_sD;
        
    const idx_t rsi = oi + shifti;
    const idx_t rsj = (kSpatialDim > 1)?(oj + shiftj):j;
    const idx_t rsk = (kSpatialDim > 2)?(ok + shiftk):k;
    
    const idx_t osizeH = i_right_border - i_left_border;
    const idx_t osizeW = j_right_border - j_left_border;
    const idx_t osizeD = k_right_border - k_left_border;
    
    if (active)
    {
        get_shifted_values<scalar_t,idx_t,kSpatialDim,padding_mode>(
                                        rsi, osizeH, input_grad_sH,
                                        rsj, osizeW, input_grad_sW,
                                        rsk, osizeD, input_grad_sD,
                                        0, 0, pass_cond,
                                        input_grad_NC, zp, _vals_array);
        *output_grad_NCHWD = compute_interpolated<scalar_t,idx_t,kSpatialDim,true>(
                                        _vals_array, di, dj, dk, pass_cond, zp);
    }
    else {
        *output_grad_NCHWD = get_shifted_value<scalar_t,idx_t,kSpatialDim,padding_mode>(
                                            rsi, osizeH, input_grad_sH,
                                            rsj, osizeW, input_grad_sW,
                                            rsk, osizeD, input_grad_sD,
                                            0, 0, pass_cond,
                                            input_grad_NC, zp);        
    }
    
}


template <typename scalar_t, typename idx_t,
          int kSpatialDim = 1,
          BIPadding padding_mode = BIPadding::Zeros,
          bool active = false>
API_INLINE void shift_forward_kernel_nhwdc(const scalar_t* const input, scalar_t* const output, 
                                           const idx_t* const weights, const scalar_t* const dweights,
                                           const idx_t n, const idx_t i, const idx_t j, const idx_t k,
                                           const idx_t sizeC, const idx_t sizeH, const idx_t sizeW, const idx_t sizeD,
                                           const idx_t input_sN, const idx_t input_sC, const idx_t input_sH, const idx_t input_sW, const idx_t input_sD,
                                           const idx_t output_sN, const idx_t output_sC, const idx_t output_sH, const idx_t output_sW, const idx_t output_sD,
                                           const idx_t weights_sC, const idx_t weights_sS, const idx_t dweights_sC, const idx_t dweights_sS,
                                           const idx_t i_left_border, const idx_t j_left_border, const idx_t k_left_border,
                                           const idx_t i_right_border, const idx_t j_right_border, const idx_t k_right_border){
    const scalar_t* input_N = input + n*input_sN;
    const scalar_t zp = static_cast<scalar_t>(0);          
    
    const idx_t oi = i - i_left_border;
    const idx_t oj = (kSpatialDim > 1) ? j - j_left_border : j; 
    const idx_t ok = (kSpatialDim > 2) ? k - k_left_border : k;
                         
    const idx_t* w_S = (kSpatialDim > 1) ? (weights+weights_sS) : nullptr;
    const idx_t* w_2S = (kSpatialDim > 2) ? (weights+2*weights_sS) : nullptr;
    const scalar_t* dw_S = (kSpatialDim > 1) ? (dweights+dweights_sS) : nullptr;
    const scalar_t* dw_2S = (kSpatialDim > 2) ? (dweights+2*dweights_sS) : nullptr;     
              
    const idx_t pass_cond_i = (1+MAX(static_cast<idx_t>(-1),MIN(i_right_border-i-1,static_cast<idx_t>(0))))*
                              (1+MAX(static_cast<idx_t>(-1),MIN(i-i_left_border, static_cast<idx_t>(0))));
    const idx_t pass_cond_j = (kSpatialDim > 1)?(1+MAX(static_cast<idx_t>(-1),MIN(j_right_border-j-1,static_cast<idx_t>(0))))*
                                                (1+MAX(static_cast<idx_t>(-1),MIN(j-j_left_border, static_cast<idx_t>(0)))):1;
    const idx_t pass_cond_k = (kSpatialDim > 2)?(1+MAX(static_cast<idx_t>(-1),MIN(k_right_border-k-1,static_cast<idx_t>(0))))*
                                                (1+MAX(static_cast<idx_t>(-1),MIN(k-k_left_border, static_cast<idx_t>(0)))):1;
    const idx_t pass_cond = pass_cond_i * pass_cond_j * pass_cond_k;
              
    scalar_t val;
    idx_t si = i;
    idx_t sj = j;
    idx_t sk = k;
    scalar_t di = zp;
    scalar_t dj = zp;
    scalar_t dk = zp;
    scalar_t *output_NHWD = output + (n*output_sN + oi*output_sH + oj*output_sW + ok*output_sD)*pass_cond;
    for (idx_t c = 0; c < sizeC; c++)
    {
        si = i - *(weights+c*weights_sC);
        if (kSpatialDim > 1) { sj = j - *(w_S+c*weights_sC); }
        if (kSpatialDim > 2) { sk = k - *(w_2S+c*weights_sC); }
        if (active)
        {
            di = *(dweights + c*dweights_sC);
            if (kSpatialDim > 1) { dj = *(dw_S+c*dweights_sC); }
            if (kSpatialDim > 2) { dk = *(dw_2S+c*dweights_sC); }
            // define array here to avoid unnessary warnings, Hope the compiler can optimize it itself
            scalar_t _vals_array[8] = {zp, zp, zp, zp, zp, zp, zp, zp};
            get_shifted_values<scalar_t,idx_t,kSpatialDim,padding_mode>(
                                          si, sizeH, input_sH,
                                          sj, sizeW, input_sW,
                                          sk, sizeD, input_sD,
                                          c, input_sC, pass_cond,
                                          input_N, zp, _vals_array);
            val = compute_interpolated<scalar_t,idx_t,kSpatialDim, false>(_vals_array, di, dj, dk, pass_cond, zp);
        }
        else {   
            val = get_shifted_value<scalar_t,idx_t,kSpatialDim,padding_mode>(
                                           si, sizeH, input_sH,
                                           sj, sizeW, input_sW,
                                           sk, sizeD, input_sD,
                                           c, input_sC, pass_cond,
                                           input_N, zp);
        }
        output_NHWD[c*output_sC] = static_cast<scalar_t>(pass_cond)*val+static_cast<scalar_t>(1-pass_cond)*(output_NHWD[c*output_sC]);
    }
}

template <typename scalar_t, typename idx_t,
          int kSpatialDim = 1,
          BIPadding padding_mode = BIPadding::Zeros,
          bool active = false>
API_INLINE void shift_backward_kernel_nhwdc(const scalar_t* const input_grad, const scalar_t* const input, 
                                            scalar_t* const output_grad,
                                            const idx_t* const weights, const scalar_t* const dweights, 
                                            scalar_t* const weights_grad,
                                            const idx_t n, const idx_t i, const idx_t j, const idx_t k,
                                            const idx_t sizeC, const idx_t sizeH, const idx_t sizeW, const idx_t sizeD,
                                            const idx_t input_grad_sN, const idx_t input_grad_sC, const idx_t input_grad_sH,
                                            const idx_t input_grad_sW, const idx_t input_grad_sD,
                                            const idx_t input_sN, const idx_t input_sC, const idx_t input_sH,
                                            const idx_t input_sW, const idx_t input_sD,
                                            const idx_t output_grad_sN, const idx_t output_grad_sC, const idx_t output_grad_sH,
                                            const idx_t output_grad_sW, const idx_t output_grad_sD,
                                            const idx_t weights_sC, const idx_t weights_sS, const idx_t dweights_sC, 
                                            const idx_t dweights_sS, const idx_t weights_grad_sC, const idx_t weights_grad_sS,
                                            const idx_t i_left_border, const idx_t j_left_border, const idx_t k_left_border,
                                            const idx_t i_right_border, const idx_t j_right_border, const idx_t k_right_border){
    const scalar_t* const input_grad_N = input_grad + n*input_grad_sN;
    const scalar_t* const input_N = input + n*input_sN;
    const scalar_t zp = static_cast<scalar_t>(0);
    scalar_t _vals_array[8] = {zp, zp, zp, zp, zp, zp, zp, zp};
    scalar_t _new_weights_grad[3] = {zp, zp, zp};
    scalar_t input_grad_NHWDC_val;
    
    idx_t shifti = 0;
    idx_t shiftj = 0;
    idx_t shiftk = 0;
    
    scalar_t di = zp;
    scalar_t dj = zp;
    scalar_t dk = zp;

    idx_t si = i;
    idx_t sj = j;
    idx_t sk = k;       
    
    const idx_t pass_cond_i = (1+MAX(static_cast<idx_t>(-1),MIN(i_right_border-i-1,static_cast<idx_t>(0))))*
                              (1+MAX(static_cast<idx_t>(-1),MIN(i-i_left_border, static_cast<idx_t>(0))));
    const idx_t pass_cond_j = (kSpatialDim > 1)?(1+MAX(static_cast<idx_t>(-1),MIN(j_right_border-j-1,static_cast<idx_t>(0))))*
                                                (1+MAX(static_cast<idx_t>(-1),MIN(j-j_left_border, static_cast<idx_t>(0)))):1;
    const idx_t pass_cond_k = (kSpatialDim > 2)?(1+MAX(static_cast<idx_t>(-1),MIN(k_right_border-k-1,static_cast<idx_t>(0))))*
                                                (1+MAX(static_cast<idx_t>(-1),MIN(k-k_left_border, static_cast<idx_t>(0)))):1;
    const idx_t pass_cond = pass_cond_i * pass_cond_j * pass_cond_k;
    
    const idx_t oi = i - i_left_border;
    const idx_t oj = (kSpatialDim > 1) ? (j - j_left_border) : j; 
    const idx_t ok = (kSpatialDim > 2) ? (k - k_left_border) : k;    
    const scalar_t* input_grad_NHWD = input_grad_N + (oi*input_grad_sH + oj*input_grad_sW + ok*input_grad_sD)*pass_cond;;
    
    idx_t rsi = oi;
    idx_t rsj = oj;
    idx_t rsk = ok;
    
    const idx_t osizeH = i_right_border - i_left_border;
    const idx_t osizeW = j_right_border - j_left_border;
    const idx_t osizeD = k_right_border - k_left_border;
                  
    const idx_t* w_S = (kSpatialDim > 1) ? (weights+weights_sS) : nullptr;
    const idx_t* w_2S = (kSpatialDim > 2) ? (weights+2*weights_sS) : nullptr;
    const scalar_t* dw_S = (kSpatialDim > 1) ? (dweights+dweights_sS) : nullptr;
    const scalar_t* dw_2S = (kSpatialDim > 2) ? (dweights+2*dweights_sS) : nullptr;
    
    scalar_t* output_grad_NHWD= output_grad + n*output_grad_sN + i*output_grad_sH + j*output_grad_sW + k*output_grad_sD;
    
    for (idx_t c = 0; c < sizeC; c++)
    {
        shifti = *(weights+c*weights_sC);
        di = *(dweights+c*dweights_sC);
        si = i - shifti;
        rsi = oi + shifti;
        if (kSpatialDim > 1) {
            shiftj = *(w_S+c*weights_sC);
            dj = *(dw_S+c*dweights_sC);
            sj = j - shiftj;
            rsj = oj + shiftj;
        }
        if (kSpatialDim > 2) {
            shiftk = *(w_2S+c*weights_sC);
            dk = *(dw_2S+c*dweights_sC);
            sk = k - shiftk;
            rsk = ok + shiftk;
        }
        // weight gradients
        get_shifted_values<scalar_t,idx_t,kSpatialDim,padding_mode>(
                                            si, sizeH, input_sH,
                                            sj, sizeW, input_sW,
                                            sk, sizeD, input_sD,
                                            c, input_sC, pass_cond,                                       
                                            input_N, zp, _vals_array);                         
        compute_weight_gradients<scalar_t,idx_t, kSpatialDim>(_vals_array, di, dj, dk, _new_weights_grad);  
        input_grad_NHWDC_val = input_grad_NHWD[c*input_grad_sC];
        ADD((weights_grad + c*weights_grad_sC),(input_grad_NHWDC_val * _new_weights_grad[0]));
        if (kSpatialDim > 1){ADD((weights_grad + weights_grad_sS + c*weights_grad_sC),(input_grad_NHWDC_val * _new_weights_grad[1]));}
        if (kSpatialDim > 2){ADD((weights_grad + 2*weights_grad_sS + c*weights_grad_sC),(input_grad_NHWDC_val * _new_weights_grad[2]));}
        
        
        // input gradient
        if (active)
        {
            get_shifted_values<scalar_t,idx_t,kSpatialDim,padding_mode>(
                                            rsi, osizeH, input_grad_sH,
                                            rsj, osizeW, input_grad_sW,
                                            rsk, osizeD, input_grad_sD,
                                            c, input_grad_sC, pass_cond,
                                            input_grad_N, zp, _vals_array);
            *(output_grad_NHWD+c*output_grad_sC) = compute_interpolated<scalar_t,idx_t,kSpatialDim,true>(
                                                                        _vals_array, di, dj, dk, pass_cond, zp);
        }
        else {
            *(output_grad_NHWD+c*output_grad_sC) = get_shifted_value<scalar_t,idx_t,kSpatialDim,padding_mode>(
                                                                        rsi, osizeH, input_grad_sH,
                                                                        rsj, osizeW, input_grad_sW,
                                                                        rsk, osizeD, input_grad_sD,
                                                                        c, input_grad_sC, pass_cond,
                                                                        input_grad_N, zp);        
        }        
    }
}


/////////QUANTIZED
template <typename scalar_t, typename idx_t,
          int kSpatialDim = 1,
          BIPadding padding_mode = BIPadding::Zeros>
API_INLINE void shift_forward_kernel_nchwd_q(const scalar_t* const input, scalar_t* const output,
                                             const idx_t* const weights,
                                             const idx_t n, const idx_t c, const idx_t i, const idx_t j, const idx_t k,
                                             const idx_t sizeH, const idx_t sizeW, const idx_t sizeD,
                                             const idx_t input_sN, const idx_t input_sC, const idx_t input_sH, 
                                             const idx_t input_sW, const idx_t input_sD,
                                             const idx_t output_sN, const idx_t output_sC, const idx_t output_sH,
                                             const idx_t output_sW, const idx_t output_sD,
                                             const idx_t weights_sC, const idx_t weights_sS,
                                             const idx_t i_left_border, const idx_t j_left_border, const idx_t k_left_border,
                                             const idx_t i_right_border, const idx_t j_right_border, const idx_t k_right_border,
                                             const scalar_t zero_point, const idx_t weights_zero_point){
    const scalar_t* const input_NC = input + n*input_sN + c*input_sC;
              
    const idx_t oi = i - i_left_border;
    const idx_t oj = (kSpatialDim > 1) ? (j - j_left_border) : j; 
    const idx_t ok = (kSpatialDim > 2) ? (k - k_left_border) : k;
    
    const idx_t si = i - *(weights+c*weights_sC) + weights_zero_point;
    const idx_t sj = (kSpatialDim > 1) ? (j - *(weights+c*weights_sC+weights_sS) + weights_zero_point) : j;
    const idx_t sk = (kSpatialDim > 2) ? (k - *(weights+c*weights_sC+2*weights_sS) + weights_zero_point) : k;           
    
    const idx_t pass_cond_i = (1+MAX(static_cast<idx_t>(-1),MIN(i_right_border-i-1,static_cast<idx_t>(0))))*
                              (1+MAX(static_cast<idx_t>(-1),MIN(i-i_left_border, static_cast<idx_t>(0))));
    const idx_t pass_cond_j = (kSpatialDim > 1)?(1+MAX(static_cast<idx_t>(-1),MIN(j_right_border-j-1,static_cast<idx_t>(0))))*
                                                (1+MAX(static_cast<idx_t>(-1),MIN(j-j_left_border, static_cast<idx_t>(0)))):1;
    const idx_t pass_cond_k = (kSpatialDim > 2)?(1+MAX(static_cast<idx_t>(-1),MIN(k_right_border-k-1,static_cast<idx_t>(0))))*
                                                (1+MAX(static_cast<idx_t>(-1),MIN(k-k_left_border, static_cast<idx_t>(0)))):1;
    const idx_t pass_cond = pass_cond_i * pass_cond_j * pass_cond_k;
    
    if (static_cast<bool>(pass_cond)) {
        scalar_t* output_NCHWD = output + n*output_sN + c*output_sC + oi*output_sH + oj*output_sW + ok*output_sD;
        *output_NCHWD = get_shifted_value<scalar_t,idx_t,kSpatialDim,padding_mode>(
                                           si, sizeH, input_sH,
                                           sj, sizeW, input_sW,
                                           sk, sizeD, input_sD,
                                           0, 0, pass_cond, 
                                           input_NC, zero_point); 
    }
}


template <typename scalar_t, typename idx_t,
          int kSpatialDim = 1,
          BIPadding padding_mode = BIPadding::Zeros>
API_INLINE void shift_forward_kernel_nhwdc_q(const scalar_t* const input, scalar_t* const output, 
                                             const idx_t* const weights,
                                             const idx_t n, const idx_t i, const idx_t j, const idx_t k,
                                             const idx_t sizeC, const idx_t sizeH, const idx_t sizeW, const idx_t sizeD,
                                             const idx_t input_sN, const idx_t input_sC, const idx_t input_sH, 
                                             const idx_t input_sW, const idx_t input_sD,
                                             const idx_t output_sN, const idx_t output_sC, const idx_t output_sH, 
                                             const idx_t output_sW, const idx_t output_sD,
                                             const idx_t weights_sC, const idx_t weights_sS,
                                             const idx_t i_left_border, const idx_t j_left_border, const idx_t k_left_border,
                                             const idx_t i_right_border, const idx_t j_right_border, const idx_t k_right_border,
                                             const scalar_t zero_point, const idx_t weights_zero_point){
    const scalar_t* input_N = input + n*input_sN;
              
    const idx_t oi = i - i_left_border;
    const idx_t oj = (kSpatialDim > 1) ? j - j_left_border : j; 
    const idx_t ok = (kSpatialDim > 2) ? k - k_left_border : k;
                 
    const idx_t* w_S = (kSpatialDim > 1) ? (weights+weights_sS) : nullptr;
    const idx_t* w_2S = (kSpatialDim > 2) ? (weights+2*weights_sS) : nullptr;
       
    
    const idx_t pass_cond_i = (1+MAX(static_cast<idx_t>(-1),MIN(i_right_border-i-1,static_cast<idx_t>(0))))*
                              (1+MAX(static_cast<idx_t>(-1),MIN(i-i_left_border, static_cast<idx_t>(0))));
    const idx_t pass_cond_j = (kSpatialDim > 1)?(1+MAX(static_cast<idx_t>(-1),MIN(j_right_border-j-1,static_cast<idx_t>(0))))*
                                                (1+MAX(static_cast<idx_t>(-1),MIN(j-j_left_border, static_cast<idx_t>(0)))):1;
    const idx_t pass_cond_k = (kSpatialDim > 2)?(1+MAX(static_cast<idx_t>(-1),MIN(k_right_border-k-1,static_cast<idx_t>(0))))*
                                                (1+MAX(static_cast<idx_t>(-1),MIN(k-k_left_border, static_cast<idx_t>(0)))):1;
    const idx_t pass_cond = pass_cond_i * pass_cond_j * pass_cond_k;
    
    if (static_cast<bool>(pass_cond)) {
        scalar_t val;
        idx_t si = i;
        idx_t sj = j;
        idx_t sk = k;
        scalar_t *output_NHWD = output + n*output_sN + oi*output_sH + oj*output_sW + ok*output_sD;
        for (idx_t c = 0; c < sizeC; c++)
        {
            si = i - *(weights+c*weights_sC) + weights_zero_point;
            if (kSpatialDim > 1){ sj = j - *(w_S+c*weights_sC) + weights_zero_point; }
            if (kSpatialDim > 2){ sk = k - *(w_2S+c*weights_sC) + weights_zero_point; }
            val = get_shifted_value<scalar_t,idx_t,kSpatialDim,padding_mode>(
                                               si, sizeH, input_sH,
                                               sj, sizeW, input_sW,
                                               sk, sizeD, input_sD,
                                               c, input_sC, pass_cond,
                                               input_N, zero_point);
            output_NHWD[c*output_sC] = val;
        }
    }
}


// Weights Init

template <typename scalar_t, typename idx_t,
          bool active = false>
API_INLINE void weight_init_kernel(const scalar_t* const src, idx_t* const iw, scalar_t* const dw,
                                   const idx_t i)
{
    iw[i] = static_cast<idx_t>(active?FLOOR(*(src+i)):ROUND(*(src+i)));
    if (active){
        dw[i] = *(src+i) - static_cast<scalar_t>(iw[i]);
    }
}
