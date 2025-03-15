#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "mat_mul.h"
#include "activation_function.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float* weights;
    float* biases;
    float* grad_weights;
    float* grad_biases;
    float* input;

    float* d_weights;
    float* d_biases;
    float* d_grad_weights;
    float* d_grad_biases;
    float* d_input;

    ActivationFunction* activation_function;
    ActivationFunction* d_activation_function;
    int input_size;
    int output_size;
} Layer;


typedef struct {
    Layer* layers;
    int num_layers;
    int output_size;
    float* output;
    float* d_output;
    char** activation_functions;

    cublasHandle_t cublas_handle;
} NeuralNetwork;

typedef struct {
    float (*loss)(float* y, float* y_true, int m, int n);
    void (*loss_derivative)(float* grad, float* y, float* y_true, int m, int n);
} LossFunction;

NeuralNetwork* create_neural_network(int* layer_sizes, char** activation_functions, int num_layers);
void forward_propagation(NeuralNetwork* nn, float* input, int m);
void free_neural_network(NeuralNetwork* nn);
void backward_propagation(NeuralNetwork* nn, float* grad_in, int m);

void forward_propagation_cuda(NeuralNetwork* nn, float* input, int m);
void backward_propagation_cuda(NeuralNetwork* nn, float* grad_in, int m);

void nn_write_weights_to_device(NeuralNetwork* nn);
void nn_write_weights_to_host(NeuralNetwork* nn);

void save_neural_network(NeuralNetwork* nn, FILE* file);
NeuralNetwork* load_neural_network(FILE* file);

#ifdef __cplusplus
}
#endif

#endif // NEURAL_NETWORK_H