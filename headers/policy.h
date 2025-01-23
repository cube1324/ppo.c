#ifndef POLICY_H
#define POLICY_H

#include "neural_network.h"
#include <math.h>

#define M_PI 3.14159265358979323846

typedef struct {
    NeuralNetwork* mu;
    float* log_std;
    int state_size;
    int action_size;

    float* input_action;
} GaussianPolicy;

GaussianPolicy* create_gaussian_policy(int* layer_sizes, ActivationFunction* activation_functions, int num_layers, float init_std);
void free_gaussian_policy(GaussianPolicy* policy);

void generate_gaussian_noise(float* out, int n);
float compute_log_prob(float* mu, float* log_std, float* action, int action_size);

void sample_action(GaussianPolicy* policy, float* state, float* action, float* log_prob, int m);
void log_prob(GaussianPolicy* policy, float* out, float* state, float* action, int m);
void log_prob_backwards(GaussianPolicy* policy, float* grad_in, float* state, float* action, int m);


#endif // POLICY_H