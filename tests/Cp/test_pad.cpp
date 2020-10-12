#include <iostream>

enum class BIPadding {Zeros, Border, Periodic, Reflect, Symmetric};

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

int main() 
{ 
    for (int i = 0; i<5;  i++){
    auto p = static_cast<BIPadding>(i);
    std::cout << i << ":  ";
    std::cout << infer_index<int>(-3, 10, p) << ", " ;
    std::cout << infer_index<int>(-2, 10, p) << ", " ;
    std::cout << infer_index<int>(-1, 10, p) << ", " ;
    std::cout << infer_index<int>(0, 10, p) << ", " ;
    std::cout << infer_index<int>(1, 10, p) << ", " ;
    std::cout << infer_index<int>(9, 10, p) << ", ";
    std::cout << infer_index<int>(10, 10, p) << ", " ;
    std::cout << infer_index<int>(11, 10, p) << ", " ;
    std::cout << infer_index<int>(12, 10, p) << ", " << std::endl <<std::endl;
    }

    return 0; 
}