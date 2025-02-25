#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

typedef struct {
    void (*activation)(float* x, int m, int n);
    void (*activation_derivative)(float* x, float* grad, int m, int n);
} ActivationFunction;


void ReLU(float* x, int m, int n);

void ReLU_derivative(float* x, float* grad, int m,  int n);

#endif // ACTIVATION_FUNCTION_H