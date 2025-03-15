#include "activation_function.h"

#include "cuda_helper.h"

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

__global__  void ReLU_kernel(float* x, int m, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m * n) {
        x[idx] = x[idx] > 0 ? x[idx] : 0;
    }
}

__global__ void ReLU_derivative_kernel(float* x, float* grad, int m,  int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m * n) {
        grad[idx] = x[idx] > 0 ? grad[idx] : 0;
    }
}

void ReLU_cuda(float* x, int m, int n){
    ReLU_kernel<<<DIVUP(m * n, BLOCK_SIZE), BLOCK_SIZE>>>(x, m, n);

    //cudaDeviceSynchronize();
    //cudaCheckErrors()
}

void ReLU_derivative_cuda(float* x, float* grad, int m,  int n){
    ReLU_derivative_kernel<<<DIVUP(m * n, BLOCK_SIZE), BLOCK_SIZE>>>(x, grad, m, n);

    //cudaDeviceSynchronize();
    //cudaCheckErrors()
}


ActivationFunction* build_activation_function(char* name) {
    ActivationFunction* activation_function = (ActivationFunction*)malloc(sizeof(ActivationFunction));

    if (strcmp(name, "relu") == 0) {
        activation_function->activation = &ReLU;
        activation_function->activation_derivative = &ReLU_derivative;
    } else {
        activation_function->activation = NULL;
        activation_function->activation_derivative = NULL;
    }

    return activation_function;
}


ActivationFunction* build_activation_function_cuda(char* name){
    ActivationFunction* activation_function = (ActivationFunction*)malloc(sizeof(ActivationFunction));

    if (strcmp(name, "relu") == 0) {
        activation_function->activation = &ReLU_cuda;
        activation_function->activation_derivative = &ReLU_derivative_cuda;
    } else {
        activation_function->activation = NULL;
        activation_function->activation_derivative = NULL;
    }

    return activation_function;
}