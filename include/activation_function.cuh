#ifndef ACTIVATION_FUNCTION_CUH
#define ACTIVATION_FUNCTION_CUH

__global__ void relu_kernel(float* out, float* in, int size);
__global__ void relu_derivative_kernel(float* out, float* in, int size);

#endif // ACTIVATION_FUNCTION_CUH