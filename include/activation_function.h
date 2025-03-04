#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    void (*activation)(float* x, int m, int n);
    void (*activation_derivative)(float* x, float* grad, int m, int n);
} ActivationFunction;

void ReLU(float* x, int m, int n);
void ReLU_derivative(float* x, float* grad, int m,  int n);

void ReLU_cuda(float* x, int m, int n);
void ReLU_derivative_cuda(float* x, float* grad, int m,  int n);

ActivationFunction* build_activation_function(char* name);
ActivationFunction* build_activation_function_cuda(char* name);

#ifdef __cplusplus
}
#endif

#endif // ACTIVATION_FUNCTION_H