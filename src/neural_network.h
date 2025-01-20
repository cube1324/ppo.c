#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mat_mul.h"

typedef struct {
    void (*activation)(float* x, int m, int n);
    void (*activation_derivative)(float* x, float* grad, int m, int n);
} ActivationFunction;

typedef struct {
    float* weights;
    float* biases;
    ActivationFunction* activation_function;

    float* grad_weights;
    float* grad_biases;
    float* input;
    int input_size;
    int output_size;
} Layer;


typedef struct {
    Layer* layers;
    int num_layers;
    float* output;
    int output_size;
} NeuralNetwork;

typedef struct {
    float (*loss)(float* y, float* y_true, int m, int n);
    void (*loss_derivative)(float* grad, float* y, float* y_true, int m, int n);
} LossFunction;

NeuralNetwork* create_neural_network(int* layer_sizes, ActivationFunction* activation_functions, int num_layers);
void forward_propagation(NeuralNetwork* nn, float* input, int m);
void free_neural_network(NeuralNetwork* nn);
void backward_propagation(NeuralNetwork* nn, LossFunction* loss, float* y_true, int m);

#endif // NEURAL_NETWORK_H