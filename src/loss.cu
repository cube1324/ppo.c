#include "loss.h"

#include "cuda_helper.h"

float mean_squared_error(float* y, float* y_true, int m, int n) {
    float loss = 0.0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            loss += pow(y_true[i * n + j] - y[i * n + j], 2);
        }
    }
    return loss / (m * n);
}


void mean_squared_error_derivative(float* grad, float* y, float* y_true, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
        // float d_loss = 0.0;
            grad[i * n + j] = 2 * (y[i * n + j] - y_true[i * n + j]) / (m * n);
        }
    }
}

__global__ void mean_squared_error_kernel(float* y, float* y_true, float* block_sums, int m, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n * m) ? pow(y_true[idx] - y[idx], 2) : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_sums[blockIdx.x] = sdata[0];
    }
}


float mean_squared_error_cuda(float* y, float* y_true, int m, int n){
    int block_size = BLOCK_SIZE < m * n ? BLOCK_SIZE : m * n;
    int n_blocks = DIVUP(m * n, block_size);

    float* d_block_sums;
    cudaMalloc(&d_block_sums, sizeof(float) * n_blocks);

    mean_squared_error_kernel<<<n_blocks, block_size, block_size * sizeof(float)>>>(y, y_true, d_block_sums, m, n);

    float block_sums[BLOCK_SIZE];
    cudaMemcpy(block_sums, d_block_sums, sizeof(float) * (DIVUP(m * n, block_size)), cudaMemcpyDeviceToHost);
    
    
    cudaCheckErrors();

    float loss = 0.0;
    for (int i = 0; i < DIVUP(m * n, block_size); i++) {
        loss += block_sums[i];
    }
    return loss / (m * n);
}

__global__ void mean_squared_error_derivative_kernel(float* grad, float* y, float* y_true, int m, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < m * n) {
        grad[idx] = 2 * (y[idx] - y_true[idx]) / (m * n);
    }
}

void mean_squared_error_derivative_cuda(float* grad, float* y, float* y_true, int m, int n){
    int block_size = BLOCK_SIZE < m * n ? BLOCK_SIZE : m * n;
    
    mean_squared_error_derivative_kernel<<<DIVUP(m * n, block_size), block_size>>>(grad, y, y_true, m, n);
    
    cudaCheckErrors();
}