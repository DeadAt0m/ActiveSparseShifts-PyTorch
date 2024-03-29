#include "../global_scope.h"
#include "interpolation.h"


enum class BIPadding {Zeros, Border, Periodic, Reflect, Symmetric};

template<typename T>
API_DEVICE API_INLINE T mod(const T a, const T b){return (b + (a % b)) % b;}

template<typename idx_t, BIPadding padding_mode = BIPadding::Zeros>
API_DEVICE API_INLINE idx_t infer_index(const idx_t index, const idx_t len){    
    bool odd_seq;
    switch (padding_mode){        
        case BIPadding::Zeros:
            return (index > len - 1)?-1:index;
        case BIPadding::Border:
            return MIN(len-1,MAX(index,static_cast<idx_t>(0)));
        case BIPadding::Periodic:
            return mod<idx_t>(index, len);
        case BIPadding::Reflect:
            odd_seq = static_cast<bool>((static_cast<idx_t>(index<0) + (ABS(index)- static_cast<idx_t>(index<0))/ (len-1)) & 1);
            return odd_seq?(len - 1 - mod<idx_t>(index, len - 1)):mod<idx_t>(index, len - 1);
        case BIPadding::Symmetric: 
            odd_seq = static_cast<bool>((static_cast<idx_t>(index<0) + (ABS(index)- static_cast<idx_t>(index<0))/ len) & 1); 
            return odd_seq?(len - 1 - mod<idx_t>(index, len)):mod<idx_t>(index, len);
        default:
            return (index > len - 1)?-1:index;
    }
}


template<typename scalar_t, typename idx_t, 
         int kSpatialDim = 1,
         BIPadding padding_mode = BIPadding::Zeros>
API_DEVICE API_INLINE scalar_t get_shifted_value(const idx_t i_shifted, const idx_t sizeH, const idx_t strideH,
                                                 const idx_t j_shifted, const idx_t sizeW, const idx_t strideW,
                                                 const idx_t k_shifted, const idx_t sizeD, const idx_t strideD,
                                                 const idx_t c, const idx_t strideC, const bool out_passcond,
                                                 const scalar_t* const array, const scalar_t zero_point){
    const idx_t tidx_i = (sizeH==1)?0:infer_index<idx_t,padding_mode>(i_shifted, sizeH);
    const idx_t pass_cond_i = static_cast<idx_t>(tidx_i>=0);
    const idx_t isH = tidx_i*strideH*pass_cond_i;

    const idx_t tidx_j = (kSpatialDim > 1)?((sizeW==1)?0:infer_index<idx_t,padding_mode>(j_shifted, sizeW)):0;
    const idx_t pass_cond_j = (kSpatialDim > 1)?(static_cast<idx_t>(tidx_j>=0)*pass_cond_i):pass_cond_i;                               
    const idx_t isW = (kSpatialDim > 1)?tidx_j*strideW*pass_cond_j:0;       
                                                 
    const idx_t tidx_k = (kSpatialDim > 2)?((sizeD==1)?0:infer_index<idx_t,padding_mode>(k_shifted, sizeD)):0;
    const idx_t pass_cond_k = (kSpatialDim > 2)?(static_cast<idx_t>(tidx_k>=0)*pass_cond_j):pass_cond_j;                               
    const idx_t isD = (kSpatialDim > 2)?tidx_k*strideD*pass_cond_k:0; 

    const bool pass_cond = static_cast<bool>(pass_cond_k)&&out_passcond;          
    return pass_cond?array[isH+isW+isD+c*strideC]:zero_point;      
}



template<typename scalar_t, typename idx_t,
         int kSpatialDim = 1,
         BIPadding padding_mode = BIPadding::Zeros>
API_DEVICE API_INLINE void get_shifted_values(const idx_t i_shifted, const idx_t sizeH, const idx_t strideH,
                                              const idx_t j_shifted, const idx_t sizeW, const idx_t strideW,
                                              const idx_t k_shifted, const idx_t sizeD, const idx_t strideD,
                                              const idx_t c, const idx_t strideC, const bool out_passcond,
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
API_DEVICE API_INLINE scalar_t rev_shift(const scalar_t diff_shift){
    return (reverse)?(static_cast<scalar_t>(1)-diff_shift):diff_shift;
}

template <typename scalar_t, typename idx_t, int kSpatialDim = 1, bool reverse = false>
API_DEVICE API_INLINE scalar_t compute_interpolated(const scalar_t* const v, const scalar_t diff_shiftH, 
                                                    const scalar_t diff_shiftW, const scalar_t diff_shiftD,
                                                    const bool pass_cond, const scalar_t zp){
    switch (kSpatialDim){        
        case 3:
            return pass_cond?interp3D(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7],
                                      rev_shift<scalar_t,reverse>(diff_shiftH), 
                                      rev_shift<scalar_t,reverse>(diff_shiftW),
                                      rev_shift<scalar_t,reverse>(diff_shiftD)):
                             zp;
        case 2:
            return pass_cond?interp2D(v[0], v[1], v[2], v[3], 
                                     rev_shift<scalar_t,reverse>(diff_shiftH), 
                                     rev_shift<scalar_t,reverse>(diff_shiftW)):
                             zp;
        default:
            return pass_cond?interp1D(v[0], v[1], rev_shift<scalar_t,reverse>(diff_shiftH)):
                             zp;
    }
}

template <typename scalar_t, typename idx_t, int kSpatialDim = 1>
API_DEVICE API_INLINE void compute_weight_gradients(const scalar_t* const v, const scalar_t diff_shiftH, const scalar_t diff_shiftW, const scalar_t diff_shiftD,
                                                    scalar_t* const output_grad, const bool pass_cond, const scalar_t zp){
    switch (kSpatialDim){        
        case 3:
            output_grad[0]=pass_cond?interp3D_dx(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7],
                                                diff_shiftW, diff_shiftD):zp;
            output_grad[1]=pass_cond?interp3D_dy(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7],
                                                diff_shiftH, diff_shiftD):zp;
            output_grad[2]=pass_cond?interp3D_dz(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7],
                                                diff_shiftH, diff_shiftW):zp;
            break;
        case 2:
            output_grad[0]=pass_cond?interp2D_dx(v[0], v[1], v[2], v[3], 
                                                 diff_shiftW):zp;
            output_grad[1]=pass_cond?interp2D_dy(v[0], v[1], v[2], v[3], 
                                                 diff_shiftH):zp;
            break;
        case 1:
            output_grad[0]=pass_cond?interp1D_dx(v[0], v[1]):zp;
            break;
    }
}

template <typename scalar_t, typename idx_t,
          int kSpatialDim = 1,
          BIPadding padding_mode = BIPadding::Zeros,
          bool active = false>
API_DEVICE API_INLINE void shift_forward_kernel_nchwd(const scalar_t* const input, scalar_t* const output,
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
     

    const bool pass_cond_i = (i >= i_left_border)&&(i < i_right_border);
    const bool pass_cond_j = (j >= j_left_border)&&(j < j_right_border);
    const bool pass_cond_k = (k >= k_left_border)&&(k < k_right_border);
    const bool pass_cond = pass_cond_i&&pass_cond_j&&pass_cond_k;

    if (pass_cond){       
        if (active)
        {   
            const scalar_t di = *(dweights + c*dweights_sC);
            const scalar_t dj = (kSpatialDim > 1) ? *(dweights + c*dweights_sC + dweights_sS) : zp;
            const scalar_t dk = (kSpatialDim > 2) ?  *(dweights + c*dweights_sC + 2*dweights_sS): zp;
            scalar_t vals_array[8] = {zp, zp, zp, zp, zp, zp, zp, zp};
            get_shifted_values<scalar_t,idx_t,kSpatialDim, padding_mode>(
                                            si, sizeH, input_sH,
                                            sj, sizeW, input_sW,
                                            sk, sizeD, input_sD,
                                            0, 0, true, 
                                            input_NC, zp, vals_array);
            *(output + n*output_sN +
                       c*output_sC +
                      oi*output_sH + 
                      oj*output_sW + 
                      ok*output_sD) = compute_interpolated<scalar_t,idx_t,kSpatialDim,false>(
                                                          vals_array, di, dj, dk,
                                                          true, zp);
        }
        else {
            *(output + n*output_sN +
                       c*output_sC +
                      oi*output_sH + 
                      oj*output_sW + 
                      ok*output_sD) = get_shifted_value<scalar_t,idx_t,kSpatialDim,padding_mode>(
                                                        si, sizeH, input_sH,
                                                        sj, sizeW, input_sW,
                                                        sk, sizeD, input_sD,
                                                        0, 0, true, 
                                                        input_NC, zp);  
        }
    }
}

template <typename scalar_t, typename idx_t,
          int kSpatialDim = 1,
          BIPadding padding_mode = BIPadding::Zeros,
          bool active = false>
API_DEVICE API_INLINE void shift_backward_kernel_nchwd(const scalar_t* const input_grad, const scalar_t* const input,  scalar_t* const output_grad,
                                                       const idx_t* const weights, const scalar_t* const dweights, scalar_t* const weights_grad,
                                                       const idx_t n, const idx_t c, const idx_t i, const idx_t j, const idx_t k,
                                                       const idx_t sizeC, const idx_t sizeH, const idx_t sizeW, const idx_t sizeD,
                                                       const idx_t input_grad_sN, const idx_t input_grad_sC, const idx_t input_grad_sH,
                                                       const idx_t input_grad_sW, const idx_t input_grad_sD,
                                                       const idx_t input_sN, const idx_t input_sC, const idx_t input_sH, 
                                                       const idx_t input_sW, const idx_t input_sD,
                                                       const idx_t output_grad_sN, const idx_t output_grad_sC, const idx_t output_grad_sH,
                                                       const idx_t output_grad_sW, const idx_t output_grad_sD,
                                                       const idx_t weights_sC, const idx_t weights_sS, 
                                                       const idx_t dweights_sC, const idx_t dweights_sS,
                                                       const idx_t weights_grad_sC, const idx_t weights_grad_sS,
                                                       const idx_t i_left_border, const idx_t j_left_border, const idx_t k_left_border,
                                                       const idx_t i_right_border, const idx_t j_right_border, const idx_t k_right_border){
    // i,j,k - from input
    const scalar_t* const input_grad_NC = input_grad + n*input_grad_sN + c*input_grad_sC;
    const scalar_t* const input_NC = input + n*input_sN + c*input_sC;
    const idx_t weights_numel = kSpatialDim * sizeC;
    const scalar_t zp = static_cast<scalar_t>(0);
       
    const idx_t shifti = *(weights+c*weights_sC);
    const idx_t shiftj = (kSpatialDim > 1)?*(weights+c*weights_sC + weights_sS):0;
    const idx_t shiftk = (kSpatialDim > 2)?*(weights + c*weights_sC + 2*weights_sS):0;
              
    const scalar_t di = *(dweights + c*dweights_sC);
    const scalar_t dj = (kSpatialDim > 1)?*(dweights + c*dweights_sC + dweights_sS):zp;
    const scalar_t dk = (kSpatialDim > 2)?*(dweights + c*dweights_sC + 2*dweights_sS):zp;

    const idx_t si = i - shifti;
    const idx_t sj = (kSpatialDim > 1) ? (j - shiftj) : j;
    const idx_t sk = (kSpatialDim > 2) ? (k - shiftk) : k;       
    
    const bool pass_cond_i = (i >= i_left_border)&&(i < i_right_border);
    const bool pass_cond_j = (j >= j_left_border)&&(j < j_right_border);
    const bool pass_cond_k = (k >= k_left_border)&&(k < k_right_border);
    const bool pass_cond = pass_cond_i&&pass_cond_j&&pass_cond_k;


    const idx_t oi = i - i_left_border;
    const idx_t oj = (kSpatialDim > 1) ? (j - j_left_border) : j; 
    const idx_t ok = (kSpatialDim > 2) ? (k - k_left_border) : k; 

    scalar_t vals_array[8] = {zp, zp, zp, zp, zp, zp, zp, zp};
    scalar_t new_weights_grad[3] = {zp, zp, zp};  
    const scalar_t input_grad_NCHWD_val = pass_cond?input_grad_NC[oi*input_grad_sH + oj*input_grad_sW + ok*input_grad_sD]:zp;
    
    // weight gradients
    get_shifted_values<scalar_t,idx_t,kSpatialDim,padding_mode>(
                                        si, sizeH, input_sH,
                                        sj, sizeW, input_sW,
                                        sk, sizeD, input_sD,
                                        0, 0, pass_cond,                                       
                                        input_NC, zp, vals_array);                         
    compute_weight_gradients<scalar_t,idx_t, kSpatialDim>(vals_array, di, dj, dk, new_weights_grad, pass_cond, zp);    
    ADD(weights_grad, c*weights_grad_sC, weights_numel, input_grad_NCHWD_val * new_weights_grad[0]);
    if (kSpatialDim > 1){ ADD(weights_grad, c*weights_grad_sC + weights_grad_sS, weights_numel, input_grad_NCHWD_val * new_weights_grad[1]); }
    if (kSpatialDim > 2){ ADD(weights_grad, c*weights_grad_sC + 2*weights_grad_sS, weights_numel, input_grad_NCHWD_val * new_weights_grad[2]); }                        
              
    // input gradient
        
    const idx_t rsi = oi + shifti;
    const idx_t rsj = (kSpatialDim > 1)?(oj + shiftj):oj;
    const idx_t rsk = (kSpatialDim > 2)?(ok + shiftk):ok;
    
    const idx_t osi = oi - shifti;
    const idx_t osj = (kSpatialDim > 1) ? (oj - shiftj) : oj;
    const idx_t osk = (kSpatialDim > 2) ? (ok - shiftk) : ok;   
              
    const idx_t osizeH = i_right_border - i_left_border;
    const idx_t osizeW = j_right_border - j_left_border;
    const idx_t osizeD = k_right_border - k_left_border;
    
    if (active)
    {
        get_shifted_values<scalar_t,idx_t,kSpatialDim,padding_mode>(
                                        osi, osizeH, input_grad_sH,
                                        osj, osizeW, input_grad_sW,
                                        osk, osizeD, input_grad_sD,
                                        0, 0, pass_cond,
                                        input_grad_NC, zp, vals_array);
         *(output_grad + n*output_grad_sN +
                         c*output_grad_sC +
                         i*output_grad_sH + 
                         j*output_grad_sW + 
                         k*output_grad_sD) = compute_interpolated<scalar_t,idx_t,kSpatialDim,false>(
                                                            vals_array, di, dj, dk, pass_cond, zp);
    }
    else {
        *(output_grad + n*output_grad_sN +
                        c*output_grad_sC +
                        i*output_grad_sH + 
                        j*output_grad_sW + 
                        k*output_grad_sD) = get_shifted_value<scalar_t,idx_t,kSpatialDim,padding_mode>(
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
API_DEVICE API_INLINE void shift_forward_kernel_nhwdc(const scalar_t* const input, scalar_t* const output, 
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
              
    const bool pass_cond_i = (i >= i_left_border)&&(i < i_right_border);
    const bool pass_cond_j = (j >= j_left_border)&&(j < j_right_border);
    const bool pass_cond_k = (k >= k_left_border)&&(k < k_right_border);
    const bool pass_cond = pass_cond_i&&pass_cond_j&&pass_cond_k;

    if (pass_cond){           
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
            si = i - *(weights+c*weights_sC);
            if (kSpatialDim > 1) { sj = j - *(w_S+c*weights_sC); }
            if (kSpatialDim > 2) { sk = k - *(w_2S+c*weights_sC); }
            if (active)
            {
                di = *(dweights + c*dweights_sC);
                if (kSpatialDim > 1) { dj = *(dw_S+c*dweights_sC); }
                if (kSpatialDim > 2) { dk = *(dw_2S+c*dweights_sC); }
                // define array here to avoid unnessary warnings, Hope the compiler can optimize it itself
                scalar_t vals_array[8] = {zp, zp, zp, zp, zp, zp, zp, zp};
                get_shifted_values<scalar_t,idx_t,kSpatialDim,padding_mode>(
                                            si, sizeH, input_sH,
                                            sj, sizeW, input_sW,
                                            sk, sizeD, input_sD,
                                            c, input_sC, true,
                                            input_N, zp, vals_array);
                val = compute_interpolated<scalar_t,idx_t,kSpatialDim, false>(vals_array, di, dj, dk, true, zp);
            }
            else {   
                val = get_shifted_value<scalar_t,idx_t,kSpatialDim,padding_mode>(
                                            si, sizeH, input_sH,
                                            sj, sizeW, input_sW,
                                            sk, sizeD, input_sD,
                                            c, input_sC, true,
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
API_DEVICE API_INLINE void shift_backward_kernel_nhwdc(const scalar_t* const input_grad, const scalar_t* const input, 
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
    const idx_t weights_numel = kSpatialDim * sizeC;
    const scalar_t zp = static_cast<scalar_t>(0);
    scalar_t vals_array[8] = {zp, zp, zp, zp, zp, zp, zp, zp};
    scalar_t new_weights_grad[3] = {zp, zp, zp};
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
    
    const bool pass_cond_i = (i >= i_left_border)&&(i < i_right_border);
    const bool pass_cond_j = (j >= j_left_border)&&(j < j_right_border);
    const bool pass_cond_k = (k >= k_left_border)&&(k < k_right_border);
    const bool pass_cond = pass_cond_i&&pass_cond_j&&pass_cond_k;
    
    const idx_t oi = i - i_left_border;
    const idx_t oj = (kSpatialDim > 1) ? (j - j_left_border) : j; 
    const idx_t ok = (kSpatialDim > 2) ? (k - k_left_border) : k;    
    const scalar_t* input_grad_NHWD = input_grad_N + (pass_cond?(oi*input_grad_sH + oj*input_grad_sW + ok*input_grad_sD):0);
    
    idx_t rsi = oi;
    idx_t rsj = oj;
    idx_t rsk = ok;
    idx_t osi = oi;
    idx_t osj = oj;
    idx_t osk = ok;
    
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
        osi = oi - shifti;
        if (kSpatialDim > 1) {
            shiftj = *(w_S+c*weights_sC);
            dj = *(dw_S+c*dweights_sC);
            sj = j - shiftj;
            rsj = oj + shiftj;
            osj = oj - shiftj;
        }
        if (kSpatialDim > 2) {
            shiftk = *(w_2S+c*weights_sC);
            dk = *(dw_2S+c*dweights_sC);
            sk = k - shiftk;
            rsk = ok + shiftk;
            osk = ok - shiftk;
        }
        // weight gradients
        get_shifted_values<scalar_t,idx_t,kSpatialDim,padding_mode>(
                                            si, sizeH, input_sH,
                                            sj, sizeW, input_sW,
                                            sk, sizeD, input_sD,
                                            c, input_sC, pass_cond,                                       
                                            input_N, zp, vals_array);                         
        compute_weight_gradients<scalar_t,idx_t, kSpatialDim>(vals_array, di, dj, dk, new_weights_grad, pass_cond, zp);  
        input_grad_NHWDC_val = input_grad_NHWD[c*input_grad_sC];
        ADD(weights_grad, c*weights_grad_sC, weights_numel, input_grad_NHWDC_val * new_weights_grad[0]);
        if (kSpatialDim > 1){ ADD(weights_grad, c*weights_grad_sC + weights_grad_sS, weights_numel, input_grad_NHWDC_val * new_weights_grad[1]); }
        if (kSpatialDim > 2){ ADD(weights_grad, c*weights_grad_sC + 2*weights_grad_sS, weights_numel, input_grad_NHWDC_val * new_weights_grad[2]); }

        
        
        // input gradient
        if (active)
        {
            get_shifted_values<scalar_t,idx_t,kSpatialDim,padding_mode>(
                                            osi, osizeH, input_grad_sH,
                                            osj, osizeW, input_grad_sW,
                                            osk, osizeD, input_grad_sD,
                                            c, input_grad_sC, pass_cond,
                                            input_grad_N, zp, vals_array);
            *(output_grad_NHWD+c*output_grad_sC) = compute_interpolated<scalar_t,idx_t,kSpatialDim,false>(
                                                                        vals_array, di, dj, dk, pass_cond, zp);
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
API_DEVICE API_INLINE void shift_forward_kernel_nchwd_q(const scalar_t* const input, scalar_t* const output,
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
    
    const bool pass_cond_i = (i >= i_left_border)&&(i < i_right_border);
    const bool pass_cond_j = (j >= j_left_border)&&(j < j_right_border);
    const bool pass_cond_k = (k >= k_left_border)&&(k < k_right_border);
    const bool pass_cond = pass_cond_i&&pass_cond_j&&pass_cond_k;
    
    if (pass_cond) {
        scalar_t* output_NCHWD = output + n*output_sN + c*output_sC + oi*output_sH + oj*output_sW + ok*output_sD;
        *output_NCHWD = get_shifted_value<scalar_t,idx_t,kSpatialDim,padding_mode>(
                                           si, sizeH, input_sH,
                                           sj, sizeW, input_sW,
                                           sk, sizeD, input_sD,
                                           0, 0, true, 
                                           input_NC, zero_point); 
    }
}


template <typename scalar_t, typename idx_t,
          int kSpatialDim = 1,
          BIPadding padding_mode = BIPadding::Zeros>
API_DEVICE API_INLINE void shift_forward_kernel_nhwdc_q(const scalar_t* const input, scalar_t* const output, 
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
       
    
    const bool pass_cond_i = (i >= i_left_border)&&(i < i_right_border);
    const bool pass_cond_j = (j >= j_left_border)&&(j < j_right_border);
    const bool pass_cond_k = (k >= k_left_border)&&(k < k_right_border);
    const bool pass_cond = pass_cond_i&&pass_cond_j&&pass_cond_k;
    
    if (pass_cond) {
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
                                               c, input_sC, true,
                                               input_N, zero_point);
            output_NHWD[c*output_sC] = val;
        }
    }
}