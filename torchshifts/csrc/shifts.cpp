#include <Python.h>
#include <torch/script.h>

#ifdef WITH_CUDA
    #include <cuda.h>
#endif

#include "shifts.h"
#include "shifts_ops.h"


#ifdef _WIN32
    #if PY_MAJOR_VERSION < 3
        PyMODINIT_FUNC init_C(void) {return NULL;}
    #else
        PyMODINIT_FUNC PyInit__C(void) {return NULL;}
    #endif
#endif

namespace shifts {
    int64_t cuda_version() {
        #ifdef WITH_CUDA
            return CUDA_VERSION;
        #else
            return -1;
        #endif
    }
} 

TORCH_LIBRARY(torchshifts, m) {
    m.def("shift1d", &shift1d);
    m.def("shift2d", &shift2d);
    m.def("shift3d", &shift3d);
    m.def("_cuda_version", &shifts::cuda_version);
}
