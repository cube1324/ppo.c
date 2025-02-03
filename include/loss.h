#ifndef LOSS_H
#define LOSS_H

#include <math.h>

float mean_squared_error(float* y, float* y_true, int m, int n);
void mean_squared_error_derivative(float* grad, float* y, float* y_true, int m, int n);

#endif // LOSS_H