
#include "mat_mul.h"
#include "cuda_helper.h"

#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <cblas.h>

#include <cublas_v2.h>

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

__global__ void add_bias_kernel(float* out, float* bias, int m, int l) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < m * l) {
        int row = idx / l;
        int col = idx % l;
        out[row * l + col] = bias[col];
    }
}

void mat_mul_cuda(cublasHandle_t handle, float* out, float* x, float* weight, float* bias, int m, int n, int l) {
    
    add_bias_kernel<<<DIVUP(m * l, BLOCK_SIZE), BLOCK_SIZE>>>(out, bias, m, l);
    
    // 2. Perform matrix multiplication using cuBLAS
    // What we want: out = x * weight^T + out
    // In column-major: this is equivalent to out^T = weight * x^T + out^T
    
    float alpha = 1.0f;
    float beta = 1.0f;  // 1.0 because we're adding to the bias already in out
    
    // Note: For row-major data with cuBLAS (which uses column-major),
    // we can swap the order of matrices and operations:
    // C = A * B in row-major is equivalent to C^T = B^T * A^T in column-major
    
    // So out = x * weight^T becomes:
    // out^T = (weight^T)^T * x^T = weight * x^T
    cublasSgemm(handle,
               CUBLAS_OP_T,      // Transpose weight
               CUBLAS_OP_N,      // Already transposed because x is row-major
               l,                // Rows of weight (now cols of out)
               m,                // Cols of x^T (now rows of out)
               n,                // Cols of weight (now rows of x^T)
               &alpha,           // alpha = 1.0
               weight,           // weight matrix
               n,                // Leading dimension of weight
               x,                // x matrix
               n,                // Leading dimension of x
               &beta,            // beta = 1.0
               out,              // out matrix
               l);               // Leading dimension of out
}

void mat_mul_backwards_cuda(cublasHandle_t handle, float* grad_x, float* grad_weight, float* grad_in, float* x, float* weight, int m, int n, int l) {
    
    float alpha = 1.0f;
    float beta = 0.0f;  // Assuming we're setting gradients, not accumulating
    
    // 1. Calculate grad_x = grad_in * weight
    // In the original code: grad_x[m,n] = grad_in[m,l] * weight[l,n]
    // For cuBLAS (column-major): this becomes (grad_x)^T = (weight)^T * (grad_in)^T
    // which is: grad_x^T[n,m] = weight^T[n,l] * grad_in^T[l,m]
    
    cublasSgemm(handle,
                CUBLAS_OP_N,     // Already transposed because they are row-major
                CUBLAS_OP_N,     // Already transposed because they are row-major
                n,               // Rows of weight^T (cols of grad_x)
                m,               // Cols of grad_in^T (rows of grad_x)
                l,               // Common dimension (cols of weight^T, rows of grad_in^T)
                &alpha,          // alpha = 1.0
                weight,          // weight matrix
                n,               // Leading dimension of weight
                grad_in,         // grad_in matrix
                l,               // Leading dimension of grad_in
                &beta,           // beta = 0.0 (set rather than accumulate)
                grad_x,          // grad_x matrix
                n);              // Leading dimension of grad_x
    
    // 2. Calculate grad_weight = grad_in^T * x
    // In original code: grad_weight[l,n] = grad_in^T[l,m] * x[m,n]
    // For cuBLAS (column-major): this becomes (grad_weight)^T = (x)^T * (grad_in)
    // which is: grad_weight^T[n,l] = x[n,m] * grad_in^T[m,l]
    
    cublasSgemm(handle,
                CUBLAS_OP_N,     // No operation on x
                CUBLAS_OP_T,     // Transpose grad_in
                n,               // Rows of x^T (cols of grad_weight)
                l,               // Cols of grad_in (rows of grad_weight)
                m,               // Common dimension (cols of x^T, rows of grad_in)
                &alpha,          // alpha = 1.0
                x,               // x matrix
                n,               // Leading dimension of x
                grad_in,         // grad_in matrix
                l,               // Leading dimension of grad_in
                &beta,           // beta = 0.0 (set rather than accumulate)
                grad_weight,     // grad_weight matrix
                n);              // Leading dimension of grad_weight

    // dim3 block_size(32, 32);
    // dim3 grid_size = dim3(DIVUP(l, block_size.x), DIVUP(n, block_size.y));

    // mat_mul_backwards_kernel_grad_weight<<<grid_size, block_size>>>(grad_weight, grad_in, x, m, n, l);
    
    // Clean up
    // cublasDestroy(handle);
}

// void mat_mul_backwards_cuda(float* grad_x, float* grad_weight, float* grad_in, float* x, float* weight, int m, int n, int l){
//     dim3 block_size(32, 32);

//     dim3 grid_size(DIVUP(m, block_size.x), DIVUP(n, block_size.y));

//     mat_mul_backwards_kernel_grad_x<<<grid_size, block_size>>>(grad_x, grad_in, weight, m, n, l);

//     grid_size = dim3(DIVUP(l, block_size.x), DIVUP(n, block_size.y));

//     mat_mul_backwards_kernel_grad_weight<<<grid_size, block_size>>>(grad_weight, grad_in, x, m, n, l);

//     //cudaDeviceSynchronize();
//     //cudaCheckErrors()
// }

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
