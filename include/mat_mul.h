// mat_mul.h
#ifndef MAT_MUL_H
#define MAT_MUL_H

#include <stdio.h>

#ifdef OMP
#include <omp.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

    void mat_mul(float* out, float* x, float* weight, float* bias, int m, int n, int l);
    void mat_mul_backwards(float* grad_x, float* grad_weight, float* grad_in, float* x, float* weight, int m, int n, int l);

    void mat_mul_cuda(float* out, float* x, float* weight, float* bias, int m, int n, int l);
    void mat_mul_backwards_cuda(float* grad_x, float* grad_weight, float* grad_in, float* x, float* weight, int m, int n, int l);

#ifdef __cplusplus
}
#endif
#endif // MAT_MUL_H