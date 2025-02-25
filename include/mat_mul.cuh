#ifndef MAT_MUL_CUH
#define MAT_MUL_CUH

__global__ void mat_mul_kernel(float* out, float* x, float* weight, float* bias, int m, int n, int l);
__global__ void mat_mul_backwards_kernel_grad_weight(float* grad_weight, float* grad_in, float* x, int m, int n, int l);
__global__ void mat_mul_backwards_kernel_grad_x(float* grad_x, float* grad_in, float* weight, int m, int n, int l);

#endif // MAT_MUL_CUH