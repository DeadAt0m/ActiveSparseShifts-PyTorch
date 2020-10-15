#include <iostream>
#include <vector>
enum class BIPadding {Zeros, Border, Periodic, Reflect, Symmetric};

#define ABS(x) std::abs(x)

template<typename T>
T mod(T a, T b){return (b + (a % b)) % b;}




template<typename idx_t>
idx_t infer_index(idx_t index, idx_t len, BIPadding padding_mode){
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
            odd_seq = ((idx_t)(out_index<0) + (ABS(out_index)-(idx_t)(out_index<0))/ (len-1)) & 1;
            out_index = mod<idx_t>(out_index, len - 1);
            if (odd_seq){out_index = len - 1 - out_index;}
            break;
        case BIPadding::Symmetric:
            odd_seq = ((idx_t)(out_index<0) + (ABS(out_index)-(idx_t)(out_index<0))/ len) & 1;
            out_index = mod<idx_t>(out_index, len);
            if (odd_seq){out_index = len - 1 - out_index;}
            break;
    }
    return out_index;
}

template<typename scalar_t, typename idx_t>
void get_shifted_value(idx_t i_shifted, idx_t sizeH, idx_t strideH,
                       idx_t j_shifted, idx_t sizeW, idx_t strideW,
                       idx_t k_shifted, idx_t sizeD, idx_t strideD,
                       idx_t c, idx_t strideC,
                       scalar_t* array, scalar_t zero_point, 
                       BIPadding padding_mode, scalar_t* output_value){
    *output_value = zero_point;
    idx_t tidx_i = -1;
    idx_t tidx_j = -1;
    idx_t tidx_k = -1;
    tidx_i = infer_index<idx_t>(i_shifted, sizeH, padding_mode);
    tidx_j = infer_index<idx_t>(j_shifted, sizeW, padding_mode);
    tidx_k = infer_index<idx_t>(k_shifted, sizeD, padding_mode); 
    if ((tidx_i>=0)&&(tidx_j>=0)&&(tidx_k>=0)){ *output_value = array[tidx_i*strideH+tidx_j*strideW+tidx_k*strideD+c*strideC];}
}



int main() 
{ 
    std::vector<int> test = {-22,-21,-20,-19,-18,-17,-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25};
    for (int i = 0; i<5;  i++){
        std::cout << i << ":  ";
        for(auto const& value: test) {
            auto p = static_cast<BIPadding>(i);
            std::cout << infer_index<int>(value, 10, p) << ", " ;
        }
    std::cout<< std::endl <<std::endl;
    }

    return 0; 
}