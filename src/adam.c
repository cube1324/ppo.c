
#include "adam.h"

Adam* create_adam(float** weights, float** grad_weights, int* length, int num_layers, int size, float beta1, float beta2) {
    Adam* adam = (Adam*)malloc(sizeof(Adam));
    adam->m = (float*)calloc(size, sizeof(float));
    adam->v = (float*)calloc(size, sizeof(float));
    adam->weights = malloc(num_layers * sizeof(float*));
    adam->grad_weights = malloc(num_layers * sizeof(float*));
    adam->lengths = malloc(num_layers * sizeof(int));
    memcpy(adam->weights, weights, num_layers * sizeof(float*));
    memcpy(adam->grad_weights, grad_weights, num_layers * sizeof(float*));
    memcpy(adam->lengths, length, num_layers * sizeof(int));
    
    adam->size = size;
    adam->beta1 = beta1;
    adam->beta2 = beta2;
    adam->time_step = 0;
    adam->num_layers = num_layers;
    return adam;
}

Adam* create_adam_from_nn(NeuralNetwork* nn, float beta1, float beta2) {
    float* weights[2 * nn->num_layers - 1];
    float* grad_weights[2 * nn->num_layers - 1];
    int length[2 * nn->num_layers - 1];

    int size = 0;
    for (int i = 0; i < nn->num_layers - 1; i++) {
        weights[i * 2] = nn->layers[i].weights;
        weights[i * 2 + 1] = nn->layers[i].biases;
        grad_weights[i * 2] = nn->layers[i].grad_weights;
        grad_weights[i * 2 + 1] = nn->layers[i].grad_biases;
        length[i * 2] = nn->layers[i].input_size * nn->layers[i].output_size;
        length[i * 2 + 1] = nn->layers[i].output_size;
        size += nn->layers[i].input_size * nn->layers[i].output_size;
        size += nn->layers[i].output_size;
    }
    return create_adam(weights, grad_weights, length, 2 * (nn->num_layers - 1), size, beta1, beta2);
}

void free_adam(Adam* adam) {
    free(adam->weights);
    free(adam->grad_weights);
    free(adam->lengths);
    free(adam->m);
    free(adam->v);
    free(adam);
}

void adam_update(Adam* adam, float lr) {
    adam->time_step = 1;

    float bias_correction1 = 1 - powf(adam->beta1, adam->time_step);
    float bias_correction2 = 1 - powf(adam->beta2, adam->time_step);

    float step_size = lr / bias_correction1;

    int current_idx = 0;
    for (int i = 0; i < adam->num_layers; i++){
        for (int j = 0; j < adam->lengths[i]; j++){
            adam->m[current_idx] = adam->beta1 * adam->m[current_idx] + (1 - adam->beta1) * adam->grad_weights[i][j];
            adam->v[current_idx] = adam->beta2 * adam->v[current_idx] + (1 - adam->beta2) * powf(adam->grad_weights[i][j], 2);
        
            float denom = sqrtf(adam->v[current_idx] / bias_correction2) + 1e-8;
            
            adam->weights[i][j] -= step_size * adam->m[current_idx] / denom;

            current_idx++;
        }
    }
}