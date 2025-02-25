#ifndef ACTIVATION_FUNCTION_CUH
#define ACTIVATION_FUNCTION_CUH

#include <cuda_runtime.h>

__global__ void relu_kernel(float* x, int m, int n);
__global__ void relu_derivative_kernel(float* x, float* grad, int m, int n);

#endif // ACTIVATION_FUNCTION_CUH