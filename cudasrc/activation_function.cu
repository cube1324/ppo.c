#include "activation_function.cuh"

__global__ void relu_kernel(float* x, int m, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m * n) {
        x[i] = x[i] > 0 ? x[i] : 0;
    }
}

__global__ void relu_derivative_kernel(float* x, float* grad, int m, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m * n) {
        grad[i] = x[i] > 0 ? grad[i] : 0;
    }
}


void ReLU(float* x, int m, int n) {
    for (int i = 0; i < m * n; i++) {
        x[i] = x[i] > 0 ? x[i] : 0;
    }
}

void ReLU_derivative(float* x, float* grad, int m,  int n) {
    for (int i = 0; i < m * n; i++) {
        grad[i] = x[i] > 0 ? grad[i] : 0;
    }
}