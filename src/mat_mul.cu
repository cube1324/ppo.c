
#include "mat_mul.h"
#include "cuda_helper.h"

#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <cblas.h>


// x = m x n
// weight = l x n
// out = m x l 
// bias = l
// computes out = x @ weight.T + bias
void mat_mul_simple(float* out, float* x, float* weight, float* bias, int m, int n, int l) {
    for (int j = 0; j < m; j++){
        for (int k = 0; k < l; k++){
            for (int p = 0; p < n; p++){
                out[j * l + k] += x[j * n + p] * weight[k * n + p];
            }
            out[j * l + k] += bias[k];
        }
    }
}

void vector_matrix_product(float* out, float* x, float* weight, float* bias, int l, int n, CBLAS_TRANSPOSE trans) {
    // Initialize output with bias
    for (int i = 0; i < l; i++) {
        out[i] = bias[i];
    }

    // Perform vector-matrix multiplication using cblas_sgemv
    cblas_sgemv(CblasRowMajor, trans, l, n, 1.0, weight, n, x, 1, 1.0, out, 1);
}


void mat_mul(float* out, float* x, float* weight, float* bias, int m, int n, int l) {
    // Initialize output with bias
    if (m == 1) {
        vector_matrix_product(out, x, weight, bias, l, n, CblasNoTrans);
        return;
    }
    
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < l; j++) {
            out[i * l + j] = bias[j];
        }
    }

    // Perform matrix multiplication using cblas_sgemm
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, l, n, 1.0, x, n, weight, n, 1.0, out, l);
}

void mat_mul_backwards(float* grad_x, float* grad_weight, float* grad_in, float* x, float* weight, int m, int n, int l) {
    // grad_x = m x n
    // for (int i = 0; i < m; i++){
    //     for (int j = 0; j < n; j++){
    //         // compute output for gradx[i * n + j]
    //         for (int k = 0; k < l; k++){
    //             grad_x[i * n + j] += grad_in[ i * l + k] * weight[k * n + j];
    //         }
    //     }
    // }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, l, 1.0, grad_in, l, weight, n, 1.0, grad_x, n);


    // // grad_weight = l x n, so need to transpose while writing to it
    // for (int i = 0; i < l; i++){
    //     for (int j = 0; j < n; j++){
    //         // Compute output for grad_weight[i * n + j]
    //         for (int k = 0; k < m; k++){
    //             grad_weight[i * n + j] += grad_in[k * l + i] * x[k * n + j];
    //         }
    //     }
    // }
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, l, n, m, 1.0, grad_in, l, x, n, 1.0, grad_weight, n);
}


__global__ void mat_mul_kernel(float* out, float* x, float* weight, float* bias, int m, int n, int l) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < m && j < l) {
        float temp = bias[j];
        for (int k = 0; k < n; k++) {
            temp += x[i * n + k] * weight[j * n + k];
        }
        out[i * l + j] = temp;
    }
}

__global__ void mat_mul_backwards_kernel_grad_x(float* grad_x, float* grad_in, float* weight, int m, int n, int l) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < m && j < n) {
        float temp = 0;
        for (int k = 0; k < l; k++) {
            temp += grad_in[i * l + k] * weight[k * n + j];
        }
        grad_x[i * n + j] = temp;
    }
}

__global__ void mat_mul_backwards_kernel_grad_weight(float* grad_weight, float* grad_in, float* x, int m, int n, int l) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < l && j < n) {
        float temp = 0;
        for (int k = 0; k < m; k++) {
            temp += grad_in[k * l + i] * x[k * n + j];
        }
        grad_weight[i * n + j] = temp;
    }
}

void mat_mul_cuda(float* out, float* x, float* weight, float* bias, int m, int n, int l){
    dim3 block_size(32, 32);

    dim3 grid_size(DIVUP(m, block_size.x), DIVUP(l, block_size.y));

    mat_mul_kernel<<<grid_size, block_size>>>(out, x, weight, bias, m, n, l);

    cudaDeviceSynchronize();
    cudaCheckErrors();
}

void mat_mul_backwards_cuda(float* grad_x, float* grad_weight, float* grad_in, float* x, float* weight, int m, int n, int l){
    dim3 block_size(32, 32);

    dim3 grid_size(DIVUP(m, block_size.x), DIVUP(n, block_size.y));

    mat_mul_backwards_kernel_grad_x<<<grid_size, block_size>>>(grad_x, grad_in, weight, m, n, l);

    grid_size = dim3(DIVUP(l, block_size.x), DIVUP(n, block_size.y));

    mat_mul_backwards_kernel_grad_weight<<<grid_size, block_size>>>(grad_weight, grad_in, x, m, n, l);

    cudaDeviceSynchronize();
    cudaCheckErrors();
}

// int main() {
//     srand(time(NULL));
//     int m = 1;
//     int n = 32;
//     int l = 32;

//     float* x = (float*)malloc(m * n * sizeof(float));
//     float* weight = (float*)malloc(l * n * sizeof(float));
//     float* bias = (float*)malloc(l * sizeof(float));
//     float* out = (float*)malloc(m * l * sizeof(float));

//     if (x == NULL || weight == NULL || bias == NULL || out == NULL) {
//         printf("Memory allocation failed\n");
//         return 1;
//     }


//     int num_runs = 10;
//     double total_time = 0.0;

//     for (int i = 0; i < num_runs; i++) {
//         // Initialize x with random values
//         for (int i = 0; i < m * n; i++) {
//             x[i] = (float)rand() / RAND_MAX;
//         }

//         // Initialize weight with random values
//         for (int i = 0; i < l * n; i++) {
//             weight[i] = (float)rand() / RAND_MAX;
//         }

//         // Initialize bias with random values
//         for (int i = 0; i < l; i++) {
//             bias[i] = (float)rand() / RAND_MAX;
//         }

//         int tic = clock();
//         mat_mul2(out, x, weight, bias, m, n, l);
//         int toc = clock();
//         total_time += (double)(toc - tic) / CLOCKS_PER_SEC;
//     }

//     double average_time = total_time / num_runs;
//     printf("Average time taken: %f\n", average_time);

//     printf("Output: [%f, %f]\n", out[2], out[3]);


//     free(x);
//     free(weight);
//     free(bias);
//     free(out);

//     return 0;
// }
