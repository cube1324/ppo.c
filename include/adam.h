#ifndef ADAM_H
#define ADAM_H

#include "neural_network.h"

typedef struct {
    float** weights;
    float** grad_weights;
    int* lengths;
    float* m;
    float* v;
    float beta1;
    float beta2;
    int time_step;
    int size;
    int num_layers;
} Adam;


Adam* create_adam(float** weights, float** grad_weights, int* length, int num_layers, int size, float beta1, float beta2);
Adam* create_adam_from_nn(NeuralNetwork* nn, float beta1, float beta2);
void free_adam(Adam* adam);

void adam_update(Adam* adam, float lr);

#endif // ADAM_H