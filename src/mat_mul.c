
#include "mat_mul.h"



// x = m x n
// weight = l x n
// out = m x l 
// bias = l
// computes out = x @ weight.T + bias
void mat_mul(float* out, float* x, float* weight, float* bias, int m, int n, int l) {
    for (int j = 0; j < m; j++){
        for (int k = 0; k < l; k++){
            for (int p = 0; p < n; p++){
                out[j * l + k] += x[j * n + p] * weight[k * n + p];
            }
            out[j * l + k] += bias[k];
        }
    }
}


void mat_mul_backwards(float* grad_x, float* grad_weight, float* grad_in, float* x, float* weight, int m, int n, int l) {
    // grad_x = m x n
    if (grad_x != NULL) {
        for (int i = 0; i < m; i++){
            for (int j = 0; j < n; j++){
                // compute output for gradx[i * n + j]
                for (int k = 0; k < l; k++){
                    grad_x[i * n + j] += grad_in[ i * l + k] * weight[k * n + j];
                }
            }
        }
    }


    // grad_weight = l x n, so need to transpose while writing to it

    for (int i = 0; i < l; i++){
        for (int j = 0; j < n; j++){
            // Compute output for grad_weight[i * n + j]
            for (int k = 0; k < m; k++){
                grad_weight[i * n + j] += grad_in[i * m + k] * x[k * n + j];
            }
        }
    }
}

// int main() {
//     int b = 1;
//     int m = 2;
//     int n = 2;
//     int l = 2;

//     float x[] = {1.0, 2.0, 3.0, 4.0};
//     float weight[] = {1.0, 2.0, 3.0, 4.0};
//     float bias[] = {1.0, 2.0};
//     float out[] = {0.0, 0.0, 0.0, 0.0};

//     mat_mul(out, x, weight, bias, b, m, n, l);

//     printf("Output: [%f, %f]\n", out[2], out[3]);

//     return 0;
// }
