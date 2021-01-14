#ifndef TORCHSHIFT
#define TORCHSHIFT


#include <cstdint>
#include "macros.h"

namespace shifts {
    API_EXPORT int64_t cuda_version();

namespace detail {
        //(Taken from torchvision)
        int64_t _cuda_version = cuda_version();

} 
} 

#endif 
