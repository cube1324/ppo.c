#ifndef LOSS_H
#define LOSS_H

#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

float mean_squared_error(float* y, float* y_true, int m, int n);
void mean_squared_error_derivative(float* grad, float* y, float* y_true, int m, int n);

float mean_squared_error_cuda(float* y, float* y_true, int m, int n);
void mean_squared_error_derivative_cuda(float* grad, float* y, float* y_true, int m, int n);

#ifdef __cplusplus
}
#endif

#endif // LOSS_H