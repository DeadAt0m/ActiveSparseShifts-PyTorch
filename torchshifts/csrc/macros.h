#pragma once
 

#ifdef TORCH18
    #define TS_TORCH_LIBRARY_FRAGMENT(ns,m) TORCH_LIBRARY_FRAGMENT(ns, m)
#else
    #define TS_TORCH_LIBRARY_FRAGMENT(ns,m) TORCH_LIBRARY_FRAGMENT_THIS_API_IS_FOR_PER_OP_REGISTRATION_ONLY(ns, m)
#endif


#ifdef _WIN32
#if defined(TORCHXIVOPS_EXPORTS)
#define API_EXPORT __declspec(dllexport)
#else
#define API_EXPORT __declspec(dllimport)
#endif
#else
#define API_EXPORT
#endif
