#ifndef CUDA_HELPER_CUH
#define CUDA_HELPER_CUH

#include <cuda_runtime.h>
#include <iostream>


#define cudaErrorCheck(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in file '%s' in line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#endif // CUDA_HELPER_CUH