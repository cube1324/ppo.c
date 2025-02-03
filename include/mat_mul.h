// mat_mul.h
#ifndef MAT_MUL_H
#define MAT_MUL_H

#include <stdio.h>

#ifdef OMP
#include <omp.h>
#endif

void mat_mul(float* out, float* x, float* weight, float* bias, int m, int n, int l);
void mat_mul_backwards(float* grad_x, float* grad_weight, float* grad_in, float* x, float* weight, int m, int n, int l);

#endif // MAT_MUL_H