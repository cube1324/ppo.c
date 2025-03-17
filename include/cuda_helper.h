#include <cuda_runtime.h>
#include <stdio.h>

#ifdef DEBUG
#define cudaCheckErrors() \
    do { \
        cudaDeviceSynchronize(); \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s at %s:%d\n", \
                cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)
#else
#define cudaCheckErrors() do { } while (0)
#endif

#define DIVUP(x, y) (((x) + (y) - 1) / (y))

#define BLOCK_SIZE 512