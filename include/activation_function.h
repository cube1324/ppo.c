#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

void ReLU(float* x, int m, int n);

void ReLU_derivative(float* x, float* grad, int m,  int n);

#endif // ACTIVATION_FUNCTION_H